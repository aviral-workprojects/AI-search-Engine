"""
Search Module — Multi-Source News Search Router
Primary: Google News RSS (free, no key)
Fallback: Tavily API
Added: Source reliability scoring (Perplexity-level)
"""

import logging
import random
import time
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse, unquote
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# USER-AGENT POOL
# ─────────────────────────────────────────────────────────────

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
]


# ─────────────────────────────────────────────────────────────
# SOURCE RELIABILITY SCORING
# ─────────────────────────────────────────────────────────────

# Tier 1 (1.0) = primary wire services / well-funded newsrooms
# Tier 2 (0.85–0.95) = established national/international outlets
# Tier 3 (0.70–0.84) = regional / specialty
# Tier 4 (0.50–0.69) = blogs, aggregators, opinion-heavy
SOURCE_WEIGHTS: Dict[str, float] = {
    # Tier 1 — Wire services & flagship outlets
    "reuters.com": 1.00,
    "apnews.com": 1.00,
    "bloomberg.com": 0.97,
    "ft.com": 0.97,

    # Tier 2 — Major international / national newsrooms
    "bbc.com": 0.95,
    "bbc.co.uk": 0.95,
    "nytimes.com": 0.95,
    "wsj.com": 0.93,
    "theguardian.com": 0.92,
    "washingtonpost.com": 0.92,
    "economist.com": 0.92,
    "cnn.com": 0.85,
    "nbcnews.com": 0.85,
    "abcnews.go.com": 0.85,
    "cbsnews.com": 0.85,
    "npr.org": 0.88,
    "politico.com": 0.87,
    "theatlantic.com": 0.87,
    "time.com": 0.85,
    "forbes.com": 0.82,

    # Technology / Business specialty
    "techcrunch.com": 0.80,
    "wired.com": 0.82,
    "arstechnica.com": 0.83,
    "theverge.com": 0.80,
    "venturebeat.com": 0.78,

    # Science
    "nature.com": 0.95,
    "sciencemag.org": 0.95,
    "newscientist.com": 0.85,

    # Tier 3 — Regional / Aggregators
    "aljazeera.com": 0.80,
    "dw.com": 0.80,
    "france24.com": 0.80,
    "ndtv.com": 0.75,
    "hindustantimes.com": 0.74,
    "timesofindia.indiatimes.com": 0.73,
    "thehindu.com": 0.78,

    # Tier 4 — Lower trust
    "medium.com": 0.60,
    "substack.com": 0.60,
    "reddit.com": 0.45,
    "quora.com": 0.40,
}

# Default score for any domain not in the table
DEFAULT_SOURCE_SCORE = 0.70


def get_source_score(url: str) -> float:
    """
    Return a reliability score [0.0, 1.0] for a given article URL.

    Lookup order:
    1. Exact domain match (e.g. 'bbc.com')
    2. Subdomain strip (e.g. 'news.bbc.co.uk' → 'bbc.co.uk')
    3. Default fallback

    Args:
        url: Article URL string.

    Returns:
        Float in [0.0, 1.0].
    """
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if domain in SOURCE_WEIGHTS:
            return SOURCE_WEIGHTS[domain]

        # Try stripping one subdomain level
        parts = domain.split(".")
        if len(parts) >= 2:
            base = ".".join(parts[-2:])
            if base in SOURCE_WEIGHTS:
                return SOURCE_WEIGHTS[base]

    except Exception:
        pass

    return DEFAULT_SOURCE_SCORE


def compute_final_score(
    rerank_score: float,
    embedding_score: float,
    url: str,
    w_rerank: float = 0.70,
    w_embedding: float = 0.20,
    w_source: float = 0.10,
) -> float:
    """
    Weighted composite score combining retrieval quality + source trust.

    Formula:
        final = 0.70 × rerank_score
              + 0.20 × embedding_score
              + 0.10 × source_score

    Args:
        rerank_score:    Cross-encoder reranker score (0–1).
        embedding_score: Cosine similarity from embeddings (0–1).
        url:             Article URL (used to look up source score).
        w_rerank:        Weight for rerank score.
        w_embedding:     Weight for embedding score.
        w_source:        Weight for source reliability.

    Returns:
        Composite float score.
    """
    source_score = get_source_score(url)
    return (w_rerank * rerank_score) + (w_embedding * embedding_score) + (w_source * source_score)


# ─────────────────────────────────────────────────────────────
# URL HELPERS
# ─────────────────────────────────────────────────────────────

def resolve_google_news_url(google_url: str) -> Optional[str]:
    """
    Resolve a Google News redirect URL to the real article URL.
    Does NOT skip on failure — returns original URL as fallback.
    """
    if "news.google.com" not in google_url:
        return google_url

    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = requests.head(
            google_url, headers=headers, allow_redirects=True, timeout=10
        )
        real_url = response.url
        return clean_url(real_url)
    except Exception as e:
        logger.warning(f"Google News redirect failed ({e}), keeping original URL")
        # Return original URL rather than None — don't silently drop articles
        return google_url


def clean_url(url: str) -> str:
    """Strip common tracking parameters from a URL."""
    tracking = {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "fbclid", "gclid", "msclkid", "_ga", "mc_cid", "mc_eid",
    }
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        clean_params = {k: v for k, v in params.items() if k not in tracking}
        if clean_params:
            qs = "&".join(f"{k}={v[0]}" for k, v in clean_params.items())
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{qs}"
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except Exception:
        return url


def extract_domain(url: str) -> str:
    """Return a human-readable source label for a URL."""
    domain_map = {
        "bbc.com": "BBC", "bbc.co.uk": "BBC",
        "cnn.com": "CNN",
        "nytimes.com": "New York Times",
        "reuters.com": "Reuters",
        "bloomberg.com": "Bloomberg",
        "theguardian.com": "The Guardian",
        "wsj.com": "Wall Street Journal",
        "ft.com": "Financial Times",
        "apnews.com": "AP News",
        "washingtonpost.com": "Washington Post",
        "npr.org": "NPR",
        "techcrunch.com": "TechCrunch",
        "wired.com": "Wired",
        "thehindu.com": "The Hindu",
        "ndtv.com": "NDTV",
    }
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if domain in domain_map:
            return domain_map[domain]
        parts = domain.split(".")
        if len(parts) >= 2:
            base = ".".join(parts[-2:])
            if base in domain_map:
                return domain_map[base]
        return domain.split(".")[0].title()
    except Exception:
        return "Unknown"


# ─────────────────────────────────────────────────────────────
# SEARCH BACKENDS
# ─────────────────────────────────────────────────────────────

def search_google_news_rss(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search Google News RSS (free, no API key required).
    Google redirect URLs are resolved; failures keep the original URL.
    """
    logger.info(f"[RSS] Searching: {query}")
    try:
        rss_url = (
            f"https://news.google.com/rss/search"
            f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        )
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        resp = requests.get(rss_url, headers=headers, timeout=15)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        articles: List[Dict] = []

        for item in root.findall(".//item")[:max_results]:
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")

            if title_el is None or link_el is None:
                continue

            google_url = link_el.text or ""
            real_url = resolve_google_news_url(google_url) or google_url

            articles.append(
                {
                    "title": title_el.text or "No title",
                    "url": real_url,
                    "published": pub_el.text if pub_el is not None else "Unknown",
                    "source": "Google News RSS",
                    "source_score": get_source_score(real_url),
                }
            )

        logger.info(f"[RSS] Found {len(articles)} articles")
        return articles

    except Exception as e:
        logger.error(f"[RSS] Search failed: {e}")
        return []


def search_tavily_api(
    query: str, api_key: str, max_results: int = 10
) -> List[Dict[str, str]]:
    """Search via Tavily API (fallback when RSS yields too few results)."""
    if not api_key:
        return []

    logger.info(f"[Tavily] Searching: {query}")
    try:
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
        }
        resp = requests.post("https://api.tavily.com/search", json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for r in data.get("results", []):
            url = clean_url(r.get("url", ""))
            articles.append(
                {
                    "title": r.get("title", "No title"),
                    "url": url,
                    "published": r.get("published_date", "Unknown"),
                    "source": "Tavily",
                    "source_score": get_source_score(url),
                }
            )

        logger.info(f"[Tavily] Found {len(articles)} articles")
        return articles

    except Exception as e:
        logger.error(f"[Tavily] Search failed: {e}")
        return []


def search_news(
    query: str,
    tavily_api_key: Optional[str] = None,
    max_results: int = 10,
) -> List[Dict[str, str]]:
    """
    Multi-source search with priority routing and source scoring.

    Priority:
        1. Google News RSS (free)
        2. Tavily API (if key provided and RSS insufficient)

    Each returned article includes 'source_score' for downstream ranking.
    """
    logger.info(f"[Search] Query: {query!r}")

    all_articles: List[Dict] = []

    # Primary — Google News RSS
    all_articles.extend(search_google_news_rss(query, max_results=max_results))

    # Fallback — Tavily
    if len(all_articles) < max_results and tavily_api_key:
        extra = search_tavily_api(
            query, tavily_api_key, max_results=max_results - len(all_articles)
        )
        all_articles.extend(extra)

    # Deduplicate by URL
    seen: set = set()
    unique: List[Dict] = []
    for a in all_articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)

    # Sort by source reliability so higher-trust outlets are processed first
    unique.sort(key=lambda x: x.get("source_score", DEFAULT_SOURCE_SCORE), reverse=True)

    logger.info(f"[Search] Returning {len(unique[:max_results])} unique articles")
    return unique[:max_results]


# ─────────────────────────────────────────────────────────────
# QUERY NORMALISATION
# ─────────────────────────────────────────────────────────────

def normalize_query(query: str) -> str:
    """Basic query cleanup (whitespace, lowercase intent detection)."""
    return query.strip()