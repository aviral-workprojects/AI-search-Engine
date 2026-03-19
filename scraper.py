"""
scraper.py -- Production-Grade Scraping Utility
Parallel + Async scraping with anti-blocking, proxy rotation, and paywall bypass.

Drop this file next to app.py. Import with:
    from scraper import scrape_parallel, scrape_async_batch, scrape_hybrid, scrape_article

Key fixes vs v1:
  - _parse_html() now calls trafilatura.extract(html_string) CORRECTLY
    (does NOT call fetch_url again -- that was the 0/10 bug)
  - Windows cp1252 crash fixed: all log calls use %s style (no Unicode literals
    in the format string itself), and the StreamHandler is reconfigured to UTF-8
  - sync fallback added inside async path so nothing is silently dropped
"""

import asyncio
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
import trafilatura
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


# ---------------------------------------------------------------
# Windows-safe logging: reconfigure StreamHandler to UTF-8 so
# Unicode characters (checkmarks, arrows) don't crash cp1252.
# ---------------------------------------------------------------
def _patch_stream_handlers():
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            try:
                if hasattr(handler.stream, "reconfigure"):
                    handler.stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

_patch_stream_handlers()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------
MIN_ARTICLE_LENGTH = 200
MAX_ARTICLE_LENGTH = 20_000

USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/16.4 Safari/605.1.15"
    ),
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

# Add real proxy strings here when needed, e.g. "http://user:pass@host:port"
PROXIES: List[Optional[str]] = [None]


# ---------------------------------------------------------------
# HEADER / PROXY HELPERS
# ---------------------------------------------------------------

def get_headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
    }


def get_proxy() -> Optional[Dict]:
    chosen = random.choice(PROXIES)
    return {"http": chosen, "https": chosen} if chosen else None


# ---------------------------------------------------------------
# PAYWALL REWRITE
# ---------------------------------------------------------------

def resolve_paywall_url(url: str) -> str:
    """Rewrite known paywalled URLs to freely accessible mirrors."""
    if "medium.com" in url:
        slug = url.replace("https://", "").replace("http://", "")
        return "https://www.textise.net/showtext.aspx?strURL=" + slug
    return url


def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------
# HTML PARSER
#
# ROOT CAUSE OF THE 0/10 BUG:
#   The old version called trafilatura.fetch_url(url) inside _parse_html,
#   making a SECOND HTTP request -- on top of the one already made by aiohttp.
#   The async fetch returns raw HTML as a string; trafilatura.extract() accepts
#   that string directly. No second fetch needed or wanted.
# ---------------------------------------------------------------

def _parse_html(url: str, html: str) -> Optional[Dict]:
    """
    Extract article text from an already-fetched HTML string.

    Waterfall:
      1. trafilatura.extract(html)   -- fast, accurate
      2. newspaper3k.set_html(html)  -- fallback, good for awkward layouts
    """
    # --- trafilatura ---
    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_recall=True,
        )
        if text and len(text.strip()) >= MIN_ARTICLE_LENGTH:
            metadata = trafilatura.extract_metadata(html)
            title = (metadata.title if metadata and metadata.title else "Article")
            return {
                "url": url,
                "text": text.strip()[:MAX_ARTICLE_LENGTH],
                "title": title,
                "source": extract_domain(url),
                "method": "trafilatura",
            }
    except Exception as exc:
        logger.debug("trafilatura.extract failed for %s: %s", url[:60], exc)

    # --- newspaper3k ---
    try:
        from newspaper import Article as NPA
        art = NPA(url)
        art.set_html(html)
        art.parse()
        if art.text and len(art.text.strip()) >= MIN_ARTICLE_LENGTH:
            return {
                "url": url,
                "text": art.text.strip()[:MAX_ARTICLE_LENGTH],
                "title": art.title or "Article",
                "source": extract_domain(url),
                "method": "newspaper3k",
            }
    except Exception as exc:
        logger.debug("newspaper3k.set_html failed for %s: %s", url[:60], exc)

    return None


# ---------------------------------------------------------------
# SYNCHRONOUS SINGLE-ARTICLE SCRAPER
# ---------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def _fetch_sync(url: str) -> Optional[str]:
    """Synchronous HTTP fetch with retry + rotating headers."""
    resp = requests.get(
        url,
        headers=get_headers(),
        proxies=get_proxy(),
        timeout=12,
        allow_redirects=True,
    )
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def scrape_article(url: str) -> Optional[Dict]:
    """
    Scrape one article with a two-step waterfall:
      1. trafilatura.fetch_url() -- handles HTTP + redirect natively
      2. requests + _parse_html  -- our own fetch with retry & anti-blocking

    Paywall rewrite is applied to step 2.
    """
    logger.debug("Scraping: %s", url[:80])

    # Step 1 -- trafilatura native fetch
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_recall=True,
            )
            if text and len(text.strip()) >= MIN_ARTICLE_LENGTH:
                metadata = trafilatura.extract_metadata(downloaded)
                title = (metadata.title if metadata and metadata.title else "Article")
                return {
                    "url": url,
                    "text": text.strip()[:MAX_ARTICLE_LENGTH],
                    "title": title,
                    "source": extract_domain(url),
                    "method": "trafilatura_fetch",
                }
    except Exception as exc:
        logger.debug("trafilatura.fetch_url failed for %s: %s", url[:60], exc)

    # Step 2 -- requests with anti-blocking + paywall rewrite
    resolved = resolve_paywall_url(url)
    try:
        html = _fetch_sync(resolved)
        if html:
            result = _parse_html(url, html)
            if result:
                return result
    except Exception as exc:
        logger.debug("requests scrape failed for %s: %s", url[:60], exc)

    return None


# ---------------------------------------------------------------
# PARALLEL SCRAPING  (ThreadPoolExecutor -- safe mode)
# ---------------------------------------------------------------

def scrape_parallel(
    article_metas: List[Dict],
    max_workers: int = 5,
) -> List[Dict]:
    """
    Scrape in parallel via threads.
    max_workers=5 keeps the concurrency low enough to avoid IP bans.
    """
    urls = [a["url"] for a in article_metas]
    n = len(urls)
    logger.info("[Parallel] Scraping %d URLs with %d workers", n, max_workers)
    t0 = time.time()

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(scrape_article, url): url for url in urls}
        for future in as_completed(future_to_url, timeout=15 * n):
            url = future_to_url[future]
            try:
                result = future.result(timeout=15)
                if result:
                    results.append(result)
                    logger.debug("[Parallel] OK   %s", url[:60])
                else:
                    logger.debug("[Parallel] EMPTY %s", url[:60])
            except Exception as exc:
                logger.debug("[Parallel] FAIL %s: %s", url[:60], exc)

    elapsed = time.time() - t0
    logger.info("[Parallel] Done: %d/%d in %.1fs", len(results), n, elapsed)
    return results


# ---------------------------------------------------------------
# ASYNC SCRAPING  (aiohttp -- high-throughput mode)
# ---------------------------------------------------------------

async def _fetch_one_async(
    session: "aiohttp.ClientSession", url: str
) -> Optional[str]:
    """Async fetch of a single URL. Returns decoded HTML string or None."""
    resolved = resolve_paywall_url(url)
    try:
        async with session.get(
            resolved,
            headers=get_headers(),
            timeout=aiohttp.ClientTimeout(total=12),
            allow_redirects=True,
            ssl=False,
        ) as resp:
            if resp.status == 200:
                raw = await resp.read()
                encoding = resp.charset or "utf-8"
                try:
                    return raw.decode(encoding, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("[Async] fetch failed %s: %s", url[:60], exc)
    return None


async def _run_async_batch(article_metas: List[Dict]) -> List[Dict]:
    """
    Core coroutine:
      1. Fetch all HTML concurrently via aiohttp
      2. Parse each with _parse_html (already-fetched string -- no extra HTTP)
      3. If parse returns None, fall back to synchronous scrape_article()
    """
    urls = [a["url"] for a in article_metas]
    connector = aiohttp.TCPConnector(limit=8, ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        html_list = await asyncio.gather(
            *[_fetch_one_async(session, url) for url in urls],
            return_exceptions=False,
        )

    results: List[Dict] = []
    for url, html in zip(urls, html_list):
        result = None

        if html:
            # Use the HTML we already fetched -- no second HTTP call
            result = _parse_html(url, html)

        if not result:
            # Async fetch failed or parse returned nothing -- sync fallback
            logger.debug("[Async] sync fallback for %s", url[:60])
            result = scrape_article(url)

        if result:
            results.append(result)

    return results


def scrape_async_batch(article_metas: List[Dict]) -> List[Dict]:
    """
    Public interface for async scraping.
    Runs the event loop in a dedicated thread so it is Streamlit-safe.
    Falls back to parallel scraping if aiohttp is not installed.
    """
    if not AIOHTTP_AVAILABLE:
        logger.warning("[Async] aiohttp not installed -- using parallel fallback")
        return scrape_parallel(article_metas)

    n = len(article_metas)
    logger.info("[Async] Scraping %d URLs asynchronously", n)
    t0 = time.time()

    import concurrent.futures

    def _run_in_new_loop() -> List[Dict]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_run_async_batch(article_metas))
        finally:
            loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(_run_in_new_loop).result(timeout=120)
    except Exception as exc:
        logger.warning("[Async] Batch failed (%s) -- parallel fallback", exc)
        results = scrape_parallel(article_metas)

    elapsed = time.time() - t0
    logger.info("[Async] Done: %d/%d in %.1fs", len(results), n, elapsed)
    return results


# ---------------------------------------------------------------
# HYBRID DISPATCHER
# ---------------------------------------------------------------

ASYNC_THRESHOLD = 8


def scrape_hybrid(
    article_metas: List[Dict],
    async_threshold: int = ASYNC_THRESHOLD,
    max_workers: int = 5,
) -> List[Dict]:
    """
    Intelligent dispatcher:
      n < threshold  --> parallel (ThreadPoolExecutor, lower overhead)
      n >= threshold --> async    (aiohttp, higher throughput)

    Both paths include a per-URL sync fallback so nothing is silently dropped.
    """
    n = len(article_metas)
    if n == 0:
        return []

    if n < async_threshold:
        logger.info(
            "[Hybrid] %d URLs -> parallel mode (threshold=%d)", n, async_threshold
        )
        return scrape_parallel(article_metas, max_workers=max_workers)
    else:
        logger.info(
            "[Hybrid] %d URLs -> async mode (threshold=%d)", n, async_threshold
        )
        return scrape_async_batch(article_metas)