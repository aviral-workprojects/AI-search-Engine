"""
Retriever Module - Embedding-based Article Retrieval
Implements semantic search using sentence transformers
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import hashlib

logger = logging.getLogger(__name__)


class ArticleRetriever:
    """
    Semantic retrieval using sentence transformers
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize retriever with embedding model
        
        Args:
            model_name: HuggingFace model name
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}
        logger.info("Embedding model loaded successfully")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Cache it
        self.embeddings_cache[text_hash] = embedding
        
        return embedding
    
    def embed_articles(self, articles: List[Dict]) -> List[np.ndarray]:
        """
        Embed multiple articles
        
        Args:
            articles: List of article dictionaries with 'text' field
        
        Returns:
            List of embedding vectors
        """
        logger.info(f"Embedding {len(articles)} articles")
        
        embeddings = []
        
        for article in articles:
            text = article.get('text', '')
            
            # Use first 1000 chars for embedding (performance)
            text_sample = text[:1000]
            
            embedding = self.embed_text(text_sample)
            embeddings.append(embedding)
        
        return embeddings
    
    def calculate_similarity(self, query_embedding: np.ndarray, 
                           doc_embeddings: List[np.ndarray]) -> List[float]:
        """
        Calculate cosine similarity between query and documents
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: List of document embedding vectors
        
        Returns:
            List of similarity scores
        """
        similarities = []
        
        for doc_emb in doc_embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append(float(similarity))
        
        return similarities
    
    def retrieve_top_k(self, query: str, articles: List[Dict], 
                       k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Retrieve top-k most relevant articles for query
        
        Args:
            query: Search query
            articles: List of article dictionaries
            k: Number of articles to retrieve
        
        Returns:
            List of (article, similarity_score) tuples, sorted by relevance
        """
        logger.info(f"Retrieving top-{k} articles for query: {query[:50]}...")
        
        if not articles:
            return []
        
        # Embed query
        query_embedding = self.embed_text(query)
        
        # Embed articles
        article_embeddings = self.embed_articles(articles)
        
        # Calculate similarities
        similarities = self.calculate_similarity(query_embedding, article_embeddings)
        
        # Combine articles with scores
        scored_articles = list(zip(articles, similarities))
        
        # Sort by similarity (descending)
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k
        top_k = scored_articles[:k]
        
        logger.info(f"Top article similarity: {top_k[0][1]:.3f}")
        logger.info(f"Lowest selected similarity: {top_k[-1][1]:.3f}")
        
        return top_k


def deduplicate_articles(articles: List[Dict], threshold: float = 0.9) -> List[Dict]:
    """
    Remove near-duplicate articles using embedding similarity
    
    Args:
        articles: List of article dictionaries
        threshold: Similarity threshold for duplicates
    
    Returns:
        Deduplicated list of articles
    """
    if len(articles) <= 1:
        return articles
    
    logger.info(f"Deduplicating {len(articles)} articles (threshold: {threshold})")
    
    # Create retriever for embeddings
    retriever = ArticleRetriever()
    
    # Embed all articles
    embeddings = retriever.embed_articles(articles)
    
    # Track which articles to keep
    keep_indices = []
    
    for i in range(len(articles)):
        is_duplicate = False
        
        # Compare with already selected articles
        for j in keep_indices:
            # Calculate similarity
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            
            if sim > threshold:
                is_duplicate = True
                logger.debug(f"Article {i} is duplicate of {j} (sim: {sim:.3f})")
                break
        
        if not is_duplicate:
            keep_indices.append(i)
    
    deduplicated = [articles[i] for i in keep_indices]
    
    logger.info(f"Kept {len(deduplicated)}/{len(articles)} unique articles")
    
    return deduplicated


def build_context(articles: List[Tuple[Dict, float]], 
                  max_chars_per_article: int = 1000,
                  max_total_tokens: int = 4000) -> str:
    """
    Build structured context from top articles
    
    Format:
    [Source 1: BBC]: Article text...
    [Source 2: Reuters]: Article text...
    
    Args:
        articles: List of (article, score) tuples
        max_chars_per_article: Max characters per article
        max_total_tokens: Approximate total token limit
    
    Returns:
        Formatted context string
    """
    logger.info(f"Building context from {len(articles)} articles")
    
    context_parts = []
    total_chars = 0
    max_total_chars = max_total_tokens * 4  # Rough estimate: 1 token ≈ 4 chars
    
    for idx, (article, score) in enumerate(articles, 1):
        # Extract article info
        title = article.get('title', 'Unknown Title')
        text = article.get('text', '')
        url = article.get('url', '')
        
        # Extract domain
        from search import extract_domain
        source = extract_domain(url)
        
        # Truncate text
        text_sample = text[:max_chars_per_article].strip()
        
        # Format
        context_part = f"[Source {idx}: {source}]\nTitle: {title}\nContent: {text_sample}\n"
        
        # Check if adding this would exceed limit
        if total_chars + len(context_part) > max_total_chars:
            logger.info(f"Context limit reached, stopping at {idx-1} articles")
            break
        
        context_parts.append(context_part)
        total_chars += len(context_part)
    
    context = "\n---\n".join(context_parts)
    
    logger.info(f"Context built: {total_chars} chars, ~{total_chars//4} tokens")
    
    return context