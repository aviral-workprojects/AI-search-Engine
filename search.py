"""
Search Module - Multi-Source News Search Router
Implements Google News RSS (primary) with API fallbacks
"""

import logging
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, parse_qs, unquote
from typing import List, Dict, Optional
import time
import random

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
]


def resolve_google_news_url(google_url: str) -> Optional[str]:
    """
    Resolve Google News redirect URL to real article URL
    
    Google News URLs format:
    https://news.google.com/rss/articles/CBMi...?oc=5
    
    Args:
        google_url: Google News redirect URL
    
    Returns:
        Real article URL or None
    """
    try:
        # If not a Google News URL, return as-is
        if 'news.google.com' not in google_url:
            return google_url
        
        # Follow redirect
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = requests.head(google_url, headers=headers, 
                                allow_redirects=True, timeout=10)
        
        real_url = response.url
        
        # Clean tracking parameters
        real_url = clean_url(real_url)
        
        logger.debug(f"Resolved: {google_url[:50]}... -> {real_url}")
        return real_url
        
    except Exception as e:
        logger.warning(f"Failed to resolve Google News URL: {e}")
        return None


def clean_url(url: str) -> str:
    """
    Remove tracking parameters and normalize URL
    
    Args:
        url: URL to clean
    
    Returns:
        Cleaned URL
    """
    try:
        parsed = urlparse(url)
        
        # Remove common tracking params
        tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', '_ga', 'mc_cid', 'mc_eid'
        }
        
        # Parse query string
        params = parse_qs(parsed.query)
        
        # Filter out tracking params
        clean_params = {k: v for k, v in params.items() if k not in tracking_params}
        
        # Rebuild query string
        if clean_params:
            query_str = '&'.join([f"{k}={v[0]}" for k, v in clean_params.items()])
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query_str}"
        else:
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        return clean_url
        
    except Exception as e:
        logger.debug(f"URL cleaning failed: {e}")
        return url


def search_google_news_rss(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search Google News RSS feed (FREE, no API key needed)
    
    Args:
        query: Search query
        max_results: Maximum results to return
    
    Returns:
        List of articles with 'title', 'url', 'published'
    """
    logger.info(f"Searching Google News RSS for: {query}")
    
    try:
        # Google News RSS URL
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = requests.get(rss_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        articles = []
        
        # Extract items
        for item in root.findall('.//item')[:max_results]:
            title_elem = item.find('title')
            link_elem = item.find('link')
            pub_date_elem = item.find('pubDate')
            
            if title_elem is not None and link_elem is not None:
                google_url = link_elem.text
                
                # Resolve Google redirect to real URL
                real_url = resolve_google_news_url(google_url)
                
                if real_url:
                    articles.append({
                        'title': title_elem.text,
                        'url': real_url,
                        'published': pub_date_elem.text if pub_date_elem is not None else "Unknown",
                        'source': 'Google News RSS'
                    })
        
        logger.info(f"Found {len(articles)} articles from Google News RSS")
        return articles
        
    except Exception as e:
        logger.error(f"Google News RSS search failed: {e}")
        return []


def search_tavily_api(query: str, api_key: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search using Tavily API (fallback option)
    
    Args:
        query: Search query
        api_key: Tavily API key
        max_results: Maximum results
    
    Returns:
        List of articles
    """
    if not api_key:
        return []
    
    logger.info(f"Searching Tavily API for: {query}")
    
    try:
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_domains": ["bbc.com", "cnn.com", "reuters.com", "nytimes.com"],
            "max_results": max_results
        }
        
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        articles = []
        for result in data.get('results', []):
            articles.append({
                'title': result.get('title', 'No title'),
                'url': clean_url(result.get('url', '')),
                'published': result.get('published_date', 'Unknown'),
                'source': 'Tavily API'
            })
        
        logger.info(f"Found {len(articles)} articles from Tavily")
        return articles
        
    except Exception as e:
        logger.error(f"Tavily API search failed: {e}")
        return []


def search_news(query: str, tavily_api_key: Optional[str] = None, 
                max_results: int = 10) -> List[Dict[str, str]]:
    """
    Multi-source search router with fallback priority
    
    Priority:
    1. Google News RSS (FREE)
    2. Tavily API (if key provided)
    
    Args:
        query: Search query
        tavily_api_key: Optional Tavily API key
        max_results: Maximum results to return
    
    Returns:
        List of unique articles
    """
    logger.info(f"Starting multi-source search for: {query}")
    
    all_articles = []
    
    # Primary: Google News RSS (FREE)
    articles_rss = search_google_news_rss(query, max_results=max_results)
    all_articles.extend(articles_rss)
    
    # If not enough results, try Tavily
    if len(all_articles) < max_results and tavily_api_key:
        articles_tavily = search_tavily_api(query, tavily_api_key, 
                                            max_results=max_results - len(all_articles))
        all_articles.extend(articles_tavily)
    
    # Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    
    for article in all_articles:
        url = article['url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
    
    logger.info(f"Total unique articles found: {len(unique_articles)}")
    return unique_articles[:max_results]


def normalize_query(query: str) -> str:
    """
    Normalize search query
    
    Handles:
    - "latest news" → adds temporal keywords
    - Remove extra whitespace
    - Basic cleanup
    
    Args:
        query: Raw query
    
    Returns:
        Normalized query
    """
    query = query.strip().lower()
    
    # Add temporal context if needed
    if 'latest' in query or 'recent' in query or 'today' in query:
        # Google News RSS automatically handles temporal ranking
        pass
    
    return query


def extract_domain(url: str) -> str:
    """
    Extract clean domain name from URL
    
    Args:
        url: Article URL
    
    Returns:
        Domain name (e.g., "BBC", "Reuters")
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Clean up
        domain = domain.replace('www.', '')
        
        # Map to readable names
        domain_map = {
            'bbc.com': 'BBC',
            'bbc.co.uk': 'BBC',
            'cnn.com': 'CNN',
            'nytimes.com': 'New York Times',
            'reuters.com': 'Reuters',
            'bloomberg.com': 'Bloomberg',
            'theguardian.com': 'The Guardian',
            'wsj.com': 'Wall Street Journal',
            'ft.com': 'Financial Times',
        }
        
        return domain_map.get(domain, domain.split('.')[0].title())
        
    except Exception:
        return "Unknown Source"