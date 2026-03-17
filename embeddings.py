"""
Embeddings Module - NVIDIA LLaMA Nemotron Embedding Integration
Supports both local HuggingFace models and API-based inference
"""

import logging
import numpy as np
from typing import List, Dict, Optional
import hashlib
from diskcache import Cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
USE_NVIDIA_API = False  # Set to True if using NVIDIA API
CACHE_DIR = Path("cache/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize embedding cache
embedding_cache = Cache(str(CACHE_DIR))


class EmbeddingModel:
    """
    Advanced embedding model with NVIDIA LLaMA Nemotron support
    Falls back to MiniLM if NVIDIA model unavailable
    """
    
    def __init__(self, model_name: str = "nvidia/llama-nemotron-embed-1b-v2",
                 fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_api: bool = False):
        """
        Initialize embedding model with fallback
        
        Args:
            model_name: Primary model (NVIDIA)
            fallback_model: Fallback model (MiniLM)
            use_api: Use API instead of local model
        """
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.use_api = use_api
        self.model = None
        self.is_nvidia = False
        
        logger.info(f"Initializing embedding model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load embedding model with fallback logic"""
        try:
            if self.use_api:
                logger.info("Using NVIDIA API for embeddings")
                self.is_nvidia = True
                # API client would be initialized here
                # from nvidia import EmbeddingClient
                # self.model = EmbeddingClient(api_key=...)
            else:
                # Try loading NVIDIA model from HuggingFace
                logger.info(f"Loading NVIDIA model: {self.model_name}")
                from sentence_transformers import SentenceTransformer
                import torch
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                try:
                    self.model = SentenceTransformer(self.model_name, device=device)
                    self.is_nvidia = True
                    logger.info(f"✓ NVIDIA model loaded successfully on {device}")
                except Exception as e:
                    logger.warning(f"NVIDIA model failed: {e}, trying fallback...")
                    self.model = SentenceTransformer(self.fallback_model, device=device)
                    self.is_nvidia = False
                    logger.info(f"✓ Fallback model loaded: {self.fallback_model}")
                    
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed single text with caching
        
        Args:
            text: Text to embed
            use_cache: Whether to use cache
        
        Returns:
            Embedding vector
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            cached = embedding_cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        
        # Cache it
        if use_cache:
            embedding_cache.set(cache_key, embedding, expire=86400)  # 24 hours
        
        return embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = 32,
                    use_cache: bool = True) -> List[np.ndarray]:
        """
        Embed multiple texts with batching and caching
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            use_cache: Whether to use cache
        
        Returns:
            List of embedding vectors
        """
        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")
        
        embeddings = []
        cache_misses = []
        cache_miss_indices = []
        
        # Check cache first
        if use_cache:
            for idx, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached = embedding_cache.get(cache_key)
                
                if cached is not None:
                    embeddings.append(cached)
                else:
                    embeddings.append(None)
                    cache_misses.append(text)
                    cache_miss_indices.append(idx)
            
            logger.info(f"Cache: {len(texts) - len(cache_misses)}/{len(texts)} hits")
        else:
            cache_misses = texts
            cache_miss_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Process cache misses in batches
        if cache_misses:
            new_embeddings = self.model.encode(
                cache_misses,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(cache_misses) > 10
            )
            
            # Insert new embeddings and cache them
            for idx, embedding in zip(cache_miss_indices, new_embeddings):
                embeddings[idx] = embedding
                
                if use_cache:
                    cache_key = self._get_cache_key(texts[idx])
                    embedding_cache.set(cache_key, embedding, expire=86400)
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed search query (alias for embed_text)
        
        Args:
            query: Query text
        
        Returns:
            Query embedding vector
        """
        logger.debug(f"Embedding query: {query[:50]}...")
        return self.embed_text(query, use_cache=True)


def calculate_cosine_similarity(query_embedding: np.ndarray,
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
    
    query_norm = np.linalg.norm(query_embedding)
    
    for doc_emb in doc_embeddings:
        doc_norm = np.linalg.norm(doc_emb)
        
        if query_norm == 0 or doc_norm == 0:
            similarities.append(0.0)
        else:
            similarity = np.dot(query_embedding, doc_emb) / (query_norm * doc_norm)
            similarities.append(float(similarity))
    
    return similarities


def load_embedding_model(model_name: str = "nvidia/llama-nemotron-embed-1b-v2",
                        use_api: bool = False) -> EmbeddingModel:
    """
    Load embedding model with caching
    
    Args:
        model_name: Model name to load
        use_api: Whether to use API
    
    Returns:
        EmbeddingModel instance
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = EmbeddingModel(model_name=model_name, use_api=use_api)
    return model