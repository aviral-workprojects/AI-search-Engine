"""
Advanced Retriever - Two-Stage Retrieval with Embedding + Reranking
Implements production-grade RAG pipeline with NVIDIA models
"""

import logging
import time
from typing import List, Dict, Tuple, Optional
import numpy as np

from embeddings import EmbeddingModel, calculate_cosine_similarity
from reranker import RerankerModel, normalize_scores

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: Embedding-based retrieval (fast, retrieve top-N)
    Stage 2: Reranking (accurate, rerank to top-K)
    """
    
    def __init__(self,
                 embedding_model: Optional[EmbeddingModel] = None,
                 reranker_model: Optional[RerankerModel] = None,
                 use_nvidia: bool = True):
        """
        Initialize advanced retriever
        
        Args:
            embedding_model: Embedding model instance
            reranker_model: Reranker model instance
            use_nvidia: Whether to use NVIDIA models
        """
        self.use_nvidia = use_nvidia
        
        # Load models
        if embedding_model is None:
            if use_nvidia:
                logger.info("Loading NVIDIA embedding model...")
                from embeddings import load_embedding_model
                self.embedding_model = load_embedding_model(
                    model_name="nvidia/llama-nemotron-embed-1b-v2"
                )
            else:
                logger.info("Loading fallback embedding model...")
                from embeddings import load_embedding_model
                self.embedding_model = load_embedding_model(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        else:
            self.embedding_model = embedding_model
        
        if reranker_model is None:
            if use_nvidia:
                logger.info("Loading NVIDIA reranker model...")
                from reranker import load_reranker_model
                self.reranker_model = load_reranker_model(
                    model_name="nvidia/llama-nemotron-rerank-1b-v2"
                )
            else:
                logger.info("Loading fallback reranker model...")
                from reranker import load_reranker_model
                self.reranker_model = load_reranker_model(
                    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
        else:
            self.reranker_model = reranker_model
        
        logger.info("Advanced retriever initialized successfully")
    
    def retrieve_and_rerank(self,
                           query: str,
                           documents: List[Dict],
                           initial_k: int = 10,
                           final_k: int = 5,
                           rerank_threshold: Optional[float] = None) -> Tuple[List[Tuple[Dict, float]], Dict]:
        """
        Two-stage retrieval: Embedding → Reranking
        
        Pipeline:
        1. Embed query
        2. Embed documents (with caching)
        3. Retrieve top-N by cosine similarity
        4. Rerank top-N using cross-encoder
        5. Return top-K
        
        Args:
            query: Search query
            documents: List of document dictionaries with 'text' field
            initial_k: Number of documents to retrieve before reranking
            final_k: Number of documents to return after reranking
            rerank_threshold: Minimum rerank score to include (optional)
        
        Returns:
            (reranked_documents, metrics)
        """
        logger.info(f"=== ADVANCED RETRIEVAL PIPELINE START ===")
        logger.info(f"Query: {query[:100]}...")
        logger.info(f"Documents: {len(documents)}, Initial K: {initial_k}, Final K: {final_k}")
        
        metrics = {
            'total_documents': len(documents),
            'initial_k': initial_k,
            'final_k': final_k,
            'embedding_time': 0,
            'retrieval_time': 0,
            'reranking_time': 0,
            'total_time': 0
        }
        
        pipeline_start = time.time()
        
        # Validate inputs
        if not documents:
            logger.warning("No documents provided")
            return [], metrics
        
        if len(documents) < initial_k:
            logger.warning(f"Fewer documents ({len(documents)}) than initial_k ({initial_k})")
            initial_k = len(documents)
        
        # Stage 1: Embedding-based retrieval
        logger.info("Stage 1: Embedding-based retrieval")
        
        # Embed query
        embed_start = time.time()
        query_embedding = self.embedding_model.embed_query(query)
        
        # Embed documents (with caching)
        doc_texts = [doc.get('text', '')[:1000] for doc in documents]  # Limit for embedding
        doc_embeddings = self.embedding_model.embed_texts(doc_texts, batch_size=32)
        
        metrics['embedding_time'] = time.time() - embed_start
        logger.info(f"✓ Embedding complete: {metrics['embedding_time']:.2f}s")
        
        # Calculate similarities
        retrieval_start = time.time()
        similarities = calculate_cosine_similarity(query_embedding, doc_embeddings)
        
        # Combine documents with scores
        doc_scores = list(zip(documents, similarities))
        
        # Sort by similarity (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top initial_k
        top_initial = doc_scores[:initial_k]
        
        metrics['retrieval_time'] = time.time() - retrieval_start
        logger.info(f"✓ Initial retrieval: {len(top_initial)} documents")
        logger.info(f"  Top embedding similarity: {top_initial[0][1]:.4f}")
        logger.info(f"  Lowest embedding similarity: {top_initial[-1][1]:.4f}")
        
        # Stage 2: Reranking
        logger.info("Stage 2: Cross-encoder reranking")
        
        rerank_start = time.time()
        
        # Extract documents for reranking
        docs_to_rerank = [doc for doc, _ in top_initial]
        
        # Rerank using cross-encoder
        reranked = self.reranker_model.rerank_documents(
            query=query,
            documents=docs_to_rerank,
            top_k=None,  # Get all scores first
            return_scores=True
        )
        
        # Apply threshold if specified
        if rerank_threshold is not None:
            logger.info(f"Applying rerank threshold: {rerank_threshold}")
            reranked = [(doc, score) for doc, score in reranked if score >= rerank_threshold]
        
        # Select final top-k
        final_results = reranked[:final_k]
        
        metrics['reranking_time'] = time.time() - rerank_start
        metrics['total_time'] = time.time() - pipeline_start
        
        logger.info(f"✓ Reranking complete: {metrics['reranking_time']:.2f}s")
        logger.info(f"  Final results: {len(final_results)} documents")
        if final_results:
            logger.info(f"  Top rerank score: {final_results[0][1]:.4f}")
            logger.info(f"  Lowest rerank score: {final_results[-1][1]:.4f}")
        
        logger.info(f"=== PIPELINE COMPLETE: {metrics['total_time']:.2f}s ===")
        
        # Store embedding model info in metrics
        metrics['embedding_model'] = 'NVIDIA' if self.embedding_model.is_nvidia else 'MiniLM'
        metrics['reranker_model'] = 'NVIDIA' if self.use_nvidia else 'MiniLM'
        
        return final_results, metrics
    
    def retrieve_only(self, query: str, documents: List[Dict],
                     top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Stage 1 only: Embedding-based retrieval without reranking
        
        Args:
            query: Search query
            documents: List of documents
            top_k: Number of results to return
        
        Returns:
            Top-k documents with embedding similarity scores
        """
        logger.info(f"Embedding-only retrieval (no reranking)")
        
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Embed documents
        doc_texts = [doc.get('text', '')[:1000] for doc in documents]
        doc_embeddings = self.embedding_model.embed_texts(doc_texts, batch_size=32)
        
        # Calculate similarities
        similarities = calculate_cosine_similarity(query_embedding, doc_embeddings)
        
        # Combine and sort
        doc_scores = list(zip(documents, similarities))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]


def build_context_advanced(ranked_documents: List[Tuple[Dict, float]],
                          max_chars_per_article: int = 1000,
                          max_total_tokens: int = 4000,
                          include_scores: bool = True) -> str:
    """
    Build context from reranked documents with scores
    
    Args:
        ranked_documents: List of (document, rerank_score) tuples
        max_chars_per_article: Max characters per article
        max_total_tokens: Approximate total token limit
        include_scores: Whether to include relevance scores in context
    
    Returns:
        Formatted context string
    """
    logger.info(f"Building context from {len(ranked_documents)} reranked documents")
    
    context_parts = []
    total_chars = 0
    max_total_chars = max_total_tokens * 4  # Rough estimate
    
    for idx, (document, rerank_score) in enumerate(ranked_documents, 1):
        # Extract document info
        title = document.get('title', 'Unknown Title')
        text = document.get('text', '')
        url = document.get('url', '')
        
        # Extract domain
        from search import extract_domain
        source = extract_domain(url)
        
        # Truncate text
        text_sample = text[:max_chars_per_article].strip()
        
        # Format with score
        if include_scores:
            context_part = f"""[Source {idx}: {source}] (Relevance: {rerank_score:.2%})
Title: {title}
Content: {text_sample}
"""
        else:
            context_part = f"""[Source {idx}: {source}]
Title: {title}
Content: {text_sample}
"""
        
        # Check limit
        if total_chars + len(context_part) > max_total_chars:
            logger.info(f"Context limit reached at {idx-1} articles")
            break
        
        context_parts.append(context_part)
        total_chars += len(context_part)
    
    context = "\n---\n".join(context_parts)
    
    logger.info(f"Context built: {total_chars} chars, ~{total_chars//4} tokens")
    
    return context