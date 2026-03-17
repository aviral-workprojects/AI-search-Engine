"""
Reranker Module - NVIDIA LLaMA Nemotron Reranking Integration
Implements cross-encoder based reranking for improved relevance
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch

logger = logging.getLogger(__name__)

# Configuration
USE_NVIDIA_RERANKER_API = False


class RerankerModel:
    """
    Advanced reranking model using NVIDIA LLaMA Nemotron
    Cross-encoder based reranking for superior relevance
    """
    
    def __init__(self, model_name: str = "nvidia/llama-nemotron-rerank-1b-v2",
                 use_api: bool = False):
        """
        Initialize reranker model
        
        Args:
            model_name: Reranker model name
            use_api: Whether to use API instead of local model
        """
        self.model_name = model_name
        self.use_api = use_api
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing reranker: {model_name} on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load reranker model"""
        try:
            if self.use_api:
                logger.info("Using NVIDIA API for reranking")
                # API client initialization
                # from nvidia import RerankClient
                # self.model = RerankClient(api_key=...)
            else:
                # Load from HuggingFace
                logger.info(f"Loading reranker model: {self.model_name}")
                
                try:
                    # Try NVIDIA model
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    self.model.to(self.device)
                    self.model.eval()
                    
                    logger.info(f"✓ NVIDIA reranker loaded successfully on {self.device}")
                    
                except Exception as e:
                    logger.warning(f"NVIDIA reranker failed: {e}, using fallback...")
                    
                    # Fallback to sentence-transformers cross-encoder
                    from sentence_transformers import CrossEncoder
                    self.model = CrossEncoder(
                        'cross-encoder/ms-marco-MiniLM-L-6-v2',
                        device=self.device
                    )
                    self.tokenizer = None  # CrossEncoder handles tokenization
                    logger.info("✓ Fallback cross-encoder loaded")
                    
        except Exception as e:
            logger.error(f"Reranker loading failed: {e}")
            raise
    
    def _score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair
        
        Args:
            query: Query text
            document: Document text
        
        Returns:
            Relevance score
        """
        if self.tokenizer is not None:
            # Using transformers (NVIDIA model)
            inputs = self.tokenizer(
                query,
                document,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = torch.sigmoid(outputs.logits[0]).item()
            
            return score
        else:
            # Using CrossEncoder
            score = self.model.predict([query, document])
            return float(score)
    
    def rerank_documents(self, query: str, documents: List[Dict],
                        top_k: Optional[int] = None,
                        return_scores: bool = True) -> List[Tuple[Dict, float]]:
        """
        Rerank documents using cross-encoder scoring
        
        Args:
            query: Query text
            documents: List of document dictionaries with 'text' field
            top_k: Number of top documents to return (None = all)
            return_scores: Whether to return scores
        
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        logger.info(f"Reranking {len(documents)} documents for query: {query[:50]}...")
        
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            text = doc.get('text', '')[:1000]  # Limit to 1000 chars for reranking
            pairs.append([query, text])
        
        # Score all pairs
        logger.debug("Computing reranker scores...")
        
        if self.tokenizer is not None:
            # Batch scoring with transformers
            scores = []
            for pair in pairs:
                score = self._score_pair(pair[0], pair[1])
                scores.append(score)
        else:
            # Batch scoring with CrossEncoder
            scores = self.model.predict(pairs)
            scores = [float(s) for s in scores]
        
        # Combine documents with scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-k
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        logger.info(f"Reranking complete. Top score: {doc_scores[0][1]:.4f}, "
                   f"Lowest: {doc_scores[-1][1]:.4f}")
        
        if return_scores:
            return doc_scores
        else:
            return [doc for doc, _ in doc_scores]
    
    def rerank_batch(self, query: str, documents: List[Dict],
                    batch_size: int = 16, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Rerank documents in batches for efficiency
        
        Args:
            query: Query text
            documents: List of documents
            batch_size: Batch size for processing
            top_k: Number of results to return
        
        Returns:
            Top-k reranked documents with scores
        """
        logger.info(f"Batch reranking {len(documents)} documents (batch_size={batch_size})")
        
        all_scores = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_scores = self.rerank_documents(query, batch_docs, return_scores=True)
            all_scores.extend(batch_scores)
        
        # Sort and return top-k
        all_scores.sort(key=lambda x: x[1], reverse=True)
        return all_scores[:top_k]


def normalize_scores(scores: List[float], method: str = "minmax") -> List[float]:
    """
    Normalize relevance scores
    
    Args:
        scores: List of scores
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    scores = np.array(scores)
    
    if method == "minmax":
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score == 0:
            return [0.5] * len(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        
    elif method == "zscore":
        mean = scores.mean()
        std = scores.std()
        
        if std == 0:
            return [0.5] * len(scores)
        
        normalized = (scores - mean) / std
        # Convert to 0-1 range
        normalized = 1 / (1 + np.exp(-normalized))
    
    else:
        return scores.tolist()
    
    return normalized.tolist()


def load_reranker_model(model_name: str = "nvidia/llama-nemotron-rerank-1b-v2",
                       use_api: bool = False) -> RerankerModel:
    """
    Load reranker model with caching
    
    Args:
        model_name: Model name to load
        use_api: Whether to use API
    
    Returns:
        RerankerModel instance
    """
    logger.info(f"Loading reranker model: {model_name}")
    model = RerankerModel(model_name=model_name, use_api=use_api)
    return model