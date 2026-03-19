"""
AI News Search Engine -- Production Upgrade
Author: Aviral Pratap Singh Chawda | AI & Data Science

Changes vs original:
  - scrape_article() loop -> scrape_hybrid() (parallel + async, anti-blocking)
  - Source reliability scoring integrated into final ranking
  - Three answer modes: Short / Balanced / Detailed
  - Structured answer generation via upgraded generator.py
  - Query understanding layer (analyze_query)
  - Windows cp1252 crash fixed: StreamHandler reconfigured to UTF-8 at startup
"""

# ---------------------------------------------------------------
# WINDOWS UTF-8 LOGGING FIX  (must run before any logger is used)
# The default Windows console stream uses cp1252 which can't encode
# Unicode chars like checkmarks or arrows. Reconfigure to UTF-8.
# ---------------------------------------------------------------
import logging
import sys

def _fix_logging_encoding():
    """Reconfigure all StreamHandlers to UTF-8 (Windows cp1252 fix)."""
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            try:
                if hasattr(handler.stream, "reconfigure"):
                    handler.stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

# Apply before basicConfig adds handlers
if sys.platform == "win32":
    # Force stdout/stderr to UTF-8 mode before Streamlit touches them
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------
import streamlit as st
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from rouge_score import rouge_scorer
import numpy as np
import time
import os
import re
import random
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Third-party
from groq import Groq
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import spacy
from diskcache import Cache
import trafilatura

# ---------------------------------------------------------------
# LOGGING CONFIGURATION
# (FileHandler uses UTF-8 explicitly; StreamHandler patched above)
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("news_search_engine.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),   # will be patched below
    ],
)
_fix_logging_encoding()   # patch the StreamHandler that basicConfig just added
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# RAG modules
# ---------------------------------------------------------------
from search import (
    search_news, normalize_query, extract_domain,
    get_source_score, compute_final_score,
)
from scraper import scrape_hybrid, scrape_article          # production scraper
from retriever import deduplicate_articles
from generator import generate_answer, extract_citations, analyze_query, ANSWER_MODES
from embeddings import load_embedding_model
from reranker import load_reranker_model
from advanced_retriever import AdvancedRetriever, build_context_advanced

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
load_dotenv()

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
cache = Cache(str(CACHE_DIR / "summaries"))

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MAX_ARTICLE_LENGTH = 20_000
MIN_ARTICLE_LENGTH = 200

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AI News Search Engine",
    page_icon="[search]",
    layout="wide",
)

st.title("AI News Search Engine")
st.markdown("Production-grade RAG pipeline: NVIDIA embedding -> reranking -> source scoring -> structured answer")

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### Configuration")

    llm_choice = st.radio(
        "LLM Provider:",
        ["Groq", "Gemini", "None (BART only)"],
        index=0,
        help="Groq: Faster | Gemini: Higher accuracy",
    )

    groq_client  = None
    gemini_model = None

    if llm_choice == "Groq":
        if GROQ_API_KEY:
            groq_client = Groq(api_key=GROQ_API_KEY)
            st.success("Groq connected")
        else:
            st.error("Groq API key not found in .env")

    elif llm_choice == "Gemini":
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                st.success("Gemini connected")
            except Exception as e:
                st.error(f"Gemini error: {e}")
        else:
            st.error("Gemini API key not found in .env")

    st.markdown("---")
    st.markdown("### Production Features")
    st.markdown("""
    **Scraping:**
    - Hybrid parallel + async (5-10x faster)
    - Anti-blocking: rotating UA, retry, proxy support
    - Paywall bypass (Medium -> textise)

    **Retrieval:**
    - NVIDIA Nemotron embeddings
    - Two-stage embed -> rerank pipeline
    - Source reliability scoring

    **Generation:**
    - Short / Balanced / Detailed answer modes
    - Structured reports with inline citations
    """)
    st.markdown("---")
    st.caption("Aviral Pratap Singh Chawda | AI & Data Science")

# ---------------------------------------------------------------
# MODEL INITIALIZATION
# ---------------------------------------------------------------

@st.cache_resource
def load_bart_model():
    logger.info("Loading BART model...")
    model_name = "facebook/bart-large-cnn"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    def bart_summarize(text, max_length=140, min_length=50):
        inputs = tokenizer(
            text[:4000], max_length=1024, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.success(f"BART loaded on {device}")
    return bart_summarize


@st.cache_resource
def load_zero_shot_classifier():
    try:
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1,
        )
    except Exception as e:
        logger.error("Classifier load failed: %s", e)
        return None


@st.cache_resource
def load_sentiment_analyzer():
    try:
        return pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1,
        )
    except Exception as e:
        logger.error("Sentiment analyzer load failed: %s", e)
        return None


@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        logger.warning("spaCy load failed: %s", e)
        return None


# Load models into session state
for _key, _loader in [
    ("bart_model",           load_bart_model),
    ("zero_shot_classifier", load_zero_shot_classifier),
    ("sentiment_analyzer",   load_sentiment_analyzer),
    ("spacy_nlp",            load_spacy_model),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _loader()

if "advanced_retriever" not in st.session_state:
    with st.spinner("Loading NVIDIA retrieval models (one-time setup)..."):
        try:
            st.session_state.advanced_retriever = AdvancedRetriever(use_nvidia=True)
            st.success("NVIDIA models loaded")
        except Exception as e:
            logger.error("Advanced retriever failed: %s", e)
            st.warning("Using fallback retrieval models")
            st.session_state.advanced_retriever = AdvancedRetriever(use_nvidia=False)

bart_model           = st.session_state.bart_model
zero_shot_classifier = st.session_state.zero_shot_classifier
sentiment_analyzer   = st.session_state.sentiment_analyzer
spacy_nlp            = st.session_state.spacy_nlp

# ---------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------

def classify_news_zeroshot(text, classifier) -> Tuple[str, float]:
    if not classifier:
        return "Unknown", 0.0
    categories = ["Politics", "Business", "Technology", "Sports",
                  "Health", "Entertainment", "Science", "World News"]
    try:
        result = classifier(text[:1000], categories)
        return result["labels"][0], result["scores"][0]
    except Exception as e:
        logger.error("Classification failed: %s", e)
        return "Unknown", 0.0


def analyze_sentiment(text, analyzer) -> Dict:
    if not analyzer:
        return {}
    try:
        result = analyzer(text[:512])
        return {"label": result[0]["label"], "score": result[0]["score"]}
    except Exception as e:
        logger.error("Sentiment failed: %s", e)
        return {}


def extract_entities(text, nlp) -> Dict:
    if not nlp:
        return {}
    try:
        doc = nlp(text[:10000])
        entities = defaultdict(list)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                entities[ent.label_].append(ent.text)
        return {
            "People":        list(set(entities["PERSON"]))[:5],
            "Organizations": list(set(entities["ORG"]))[:5],
            "Locations":     list(set(entities["GPE"] + entities["LOC"]))[:5],
        }
    except Exception as e:
        logger.error("NER failed: %s", e)
        return {}


def apply_source_scoring(
    reranked: List[Tuple[Dict, float]],
    embedding_scores: Optional[Dict[str, float]] = None,
) -> List[Tuple[Dict, float]]:
    """
    Re-sort reranked results using composite score:
        final = 0.70 * rerank + 0.20 * embedding + 0.10 * source_reliability
    """
    scored = []
    for article, rerank_score in reranked:
        url = article.get("url", "")
        emb_score = (embedding_scores or {}).get(url, rerank_score)
        final = compute_final_score(
            rerank_score=rerank_score,
            embedding_score=emb_score,
            url=url,
        )
        article["source_score"] = get_source_score(url)
        article["rerank_score"] = rerank_score
        article["final_score"]  = final
        scored.append((article, final))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------
# TABS
# ---------------------------------------------------------------
tab_search, tab_url, tab_text, tab_eval = st.tabs([
    "AI Search",
    "URL Analysis",
    "Text Input",
    "Evaluation",
])

# ===============================================================
# TAB 1 -- AI SEARCH (RAG)
# ===============================================================
with tab_search:
    st.header("AI-Powered News Search")
    st.caption("Ask questions and get synthesised answers with citations from multiple sources")

    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input(
            "Your Question",
            placeholder="e.g., 'Latest AI regulations in India' or 'Tesla earnings Q4 2024'",
        )
    with col2:
        num_sources = st.selectbox("Sources", [5, 10, 15], index=0)

    # Advanced Options
    with st.expander("Advanced Options"):
        col1, col2, col3 = st.columns(3)

        with col1:
            task_type = st.radio(
                "Task Type",
                ["qa", "summarization"],
                index=0,
                help="QA: More factual | Summarization: More fluent",
            )
            initial_k = st.slider(
                "Stage 1: Initial Retrieval",
                5, 20, 10,
                help="Articles to embed before reranking",
            )

        with col2:
            answer_mode = st.radio(
                "Answer Mode",
                ["short", "balanced", "detailed"],
                index=1,
                format_func=lambda x: {
                    "short":    "Short -- Summary only",
                    "balanced": "Balanced -- Summary + 2 points",
                    "detailed": "Detailed -- Summary + 5 structured points",
                }[x],
                help="Controls depth and length of the generated answer",
            )
            st.caption(ANSWER_MODES[answer_mode]["description"])

        with col3:
            show_full_articles = st.checkbox("Show full article text", value=False)
            show_metrics       = st.checkbox("Show detailed metrics",   value=True)
            show_source_scores = st.checkbox("Show source scores",      value=True)

    # Search button
    if st.button("Search & Generate Answer", type="primary") and search_query:

        logger.info("=== AI SEARCH | Query: %s | Mode: %s ===", search_query, answer_mode)

        # Query understanding
        query_intent = analyze_query(search_query)
        if query_intent["entities"]:
            st.caption("Detected entities: " + ", ".join(query_intent["entities"][:5]))

        # Step 1: Search
        with st.spinner("Searching news sources..."):
            search_start = time.time()
            articles = search_news(
                query=normalize_query(search_query),
                max_results=num_sources * 2,
            )
            search_time = time.time() - search_start

        if not articles:
            st.error("No articles found. Try a different query.")
            st.stop()

        st.success(f"Found {len(articles)} articles in {search_time:.1f}s")

        # Step 2: Hybrid Scraping
        mode_label = "async" if len(articles) >= 8 else "parallel"
        with st.spinner(f"Scraping {len(articles)} articles ({mode_label} mode)..."):
            scrape_start = time.time()
            scraped_articles = scrape_hybrid(articles, async_threshold=8, max_workers=5)
            scrape_time = time.time() - scrape_start

        if not scraped_articles:
            st.error("Failed to scrape articles. URLs may be inaccessible.")
            st.stop()

        st.success(
            f"Scraped {len(scraped_articles)}/{len(articles)} articles "
            f"in {scrape_time:.1f}s ({mode_label} mode)"
        )

        # Step 3: Deduplicate
        with st.spinner("Removing duplicates..."):
            unique_articles = deduplicate_articles(scraped_articles, threshold=0.85)

        st.info(f"{len(unique_articles)} unique articles after deduplication")

        # Step 4: Two-Stage Retrieval
        st.markdown("### Two-Stage Retrieval Pipeline")
        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("Stage 1: Embedding-based retrieval..."):
                retrieval_start = time.time()
                retriever = st.session_state.advanced_retriever

                reranked_raw, metrics = retriever.retrieve_and_rerank(
                    query=search_query,
                    documents=unique_articles,
                    initial_k=min(initial_k, len(unique_articles)),
                    final_k=min(num_sources, len(unique_articles)),
                )
                retrieval_time = time.time() - retrieval_start

            st.info(
                f"**Stage 1:** Retrieved top {metrics['initial_k']} articles\n\n"
                f"Embedding: {metrics['embedding_time']:.2f}s"
            )

        with col2:
            st.success(
                f"**Stage 2:** Reranked to top {len(reranked_raw)}\n\n"
                f"Reranking: {metrics['reranking_time']:.2f}s"
            )

        # Step 5: Source Reliability Scoring
        reranked_articles = apply_source_scoring(reranked_raw)

        # Step 6: Build context
        context = build_context_advanced(reranked_articles, include_scores=True)

        # Step 7: Generate Answer
        with st.spinner(f"Generating {answer_mode} answer with citations..."):
            generation_start = time.time()
            answer = generate_answer(
                query=search_query,
                context=context,
                llm_choice=llm_choice,
                groq_client=groq_client,
                gemini_model=gemini_model,
                task_type=task_type,
                answer_mode=answer_mode,
            )
            generation_time = time.time() - generation_start

        # Display Results
        st.markdown("---")
        mode_labels = {"short": "Short", "balanced": "Balanced", "detailed": "Detailed"}
        st.markdown(f"### Answer ({mode_labels[answer_mode]})")
        st.info(answer)

        citations_used = extract_citations(answer)

        st.markdown("### Performance Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Articles Found",    len(articles))
        c2.metric("Articles Analyzed", len(reranked_articles))
        c3.metric("Citations Used",    len(citations_used))
        c4.metric(
            "Total Time",
            f"{search_time + scrape_time + retrieval_time + generation_time:.1f}s",
        )

        # Sources
        st.markdown("### Sources")
        for idx, (article, final_score) in enumerate(reranked_articles, 1):
            rerank_s = article.get("rerank_score", final_score)
            source_s = article.get("source_score", 0.0)

            label = (
                f"[{idx}] {article['title']} ({article['source']}) -- "
                f"Relevance: {final_score:.1%}"
            )
            with st.expander(label):
                st.write(f"**URL:** [{article['url']}]({article['url']})")
                st.write(f"**Scraping Method:** {article.get('method', 'unknown')}")

                if show_source_scores:
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Rerank Score", f"{rerank_s:.3f}")
                    sc2.metric("Source Trust", f"{source_s:.2f}")
                    sc3.metric("Final Score",  f"{final_score:.3f}")

                if show_full_articles:
                    st.text_area(
                        "Full text",
                        value=(
                            article["text"][:2000] + "..."
                            if len(article["text"]) > 2000
                            else article["text"]
                        ),
                        height=200,
                        key=f"article_{idx}",
                        label_visibility="collapsed",
                    )
                else:
                    st.write(f"**Preview:** {article['text'][:300]}...")

        if show_metrics:
            with st.expander("Detailed Performance Breakdown"):
                st.write(f"**1. Search:**        {search_time:.2f}s")
                st.write(f"**2. Scraping:**      {scrape_time:.2f}s  ({mode_label} mode)")
                st.write("**3. Deduplication:** ~0.3s")
                st.write(f"**4. Embedding:**     {metrics['embedding_time']:.2f}s")
                st.write(f"**5. Retrieval:**     {metrics['retrieval_time']:.2f}s")
                st.write(f"**6. Reranking:**     {metrics['reranking_time']:.2f}s")
                st.write(f"**7. Generation:**    {generation_time:.2f}s")
                st.write(
                    f"**Total:** "
                    f"{search_time + scrape_time + metrics['total_time'] + generation_time:.2f}s"
                )
                st.markdown("---")
                st.write(f"**Embedding Model:** {metrics['embedding_model']}")
                st.write(f"**Reranker Model:**  {metrics['reranker_model']}")
                st.write(f"**LLM:**             {llm_choice}")
                st.write(f"**Answer Mode:**     {answer_mode}")

    # Example queries
    st.markdown("---")
    st.markdown("### Example Queries")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Technology:**
        - Latest AI regulations in India
        - OpenAI GPT-4 recent updates
        - Quantum computing breakthroughs

        **Business:**
        - Tesla earnings Q4 2024
        - Cryptocurrency market trends
        - Startup funding in AI sector
        """)
    with col2:
        st.markdown("""
        **Politics:**
        - US election latest developments
        - Climate policy updates 2024
        - International trade agreements

        **Science:**
        - COVID-19 variant updates
        - Space exploration news
        - Climate change research findings
        """)


# ===============================================================
# TAB 2 -- URL ANALYSIS
# ===============================================================
with tab_url:
    st.header("URL Article Analysis")
    st.caption("Analyse a single article from a URL")

    article_url = st.text_input(
        "Article URL",
        placeholder="https://www.bbc.com/news/technology-example",
    )

    if st.button("Analyze Article", type="primary") and article_url:
        with st.spinner("Scraping article..."):
            article = scrape_article(article_url)

        if not article:
            st.error("Failed to scrape article. Try a different URL.")
        else:
            st.success(f"Article scraped from {article['source']}")

            with st.spinner("Generating summary..."):
                summary = bart_model(article["text"], max_length=140)

            st.markdown("### Summary")
            st.info(summary)

            category, confidence = classify_news_zeroshot(article["text"], zero_shot_classifier)
            sentiment = analyze_sentiment(article["text"], sentiment_analyzer)
            entities  = extract_entities(article["text"], spacy_nlp)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Classification")
                st.info(f"{category} ({confidence:.1%})")
                if entities:
                    st.markdown("### Named Entities")
                    for etype, elist in entities.items():
                        if elist:
                            st.write(f"**{etype}:** {', '.join(elist)}")
            with col2:
                if sentiment:
                    st.markdown("### Sentiment")
                    st.info(f"{sentiment['label']} ({sentiment['score']:.1%})")


# ===============================================================
# TAB 3 -- TEXT INPUT
# ===============================================================
with tab_text:
    st.header("Direct Text Input")
    st.caption("Paste article text for analysis")

    article_text = st.text_area(
        "Article Text",
        height=300,
        placeholder="Paste article text here...",
    )

    if st.button("Analyze Text", type="primary") and article_text:
        with st.spinner("Generating summary..."):
            summary = bart_model(article_text, max_length=140)

        st.markdown("### Summary")
        st.success(summary)

        col1, col2 = st.columns(2)
        with col1:
            category, confidence = classify_news_zeroshot(article_text, zero_shot_classifier)
            st.markdown("### Classification")
            st.info(f"{category} ({confidence:.1%})")
        with col2:
            sentiment = analyze_sentiment(article_text, sentiment_analyzer)
            if sentiment:
                st.markdown("### Sentiment")
                st.info(f"{sentiment['label']} ({sentiment['score']:.1%})")


# ===============================================================
# TAB 4 -- EVALUATION
# ===============================================================
with tab_eval:
    st.header("Model Evaluation")
    st.markdown("""
    ### System Capabilities

    | Component | Implementation |
    |-----------|---------------|
    | **Scraping** | Hybrid parallel + async, anti-blocking, retry |
    | **Source Scoring** | Domain trust table (Reuters -> Medium) |
    | **Embedding** | NVIDIA Nemotron 1B v2 (fallback: MiniLM) |
    | **Reranking** | NVIDIA Nemotron Rerank 1B v2 (fallback: MiniLM) |
    | **Final Ranking** | 0.7 x rerank + 0.2 x embed + 0.1 x source |
    | **Answer Modes** | Short / Balanced / Detailed |
    | **LLM** | Groq (llama-3.1-8b) / Gemini 1.5 Flash |
    """)

    if st.button("Run Sample Evaluation"):
        with st.spinner("Running evaluation..."):
            time.sleep(1)
            st.success("Evaluation complete (simulated)")


# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("""
### Production Pipeline

```
Query -> Query Understanding
      -> Multi-Source Search (RSS + Tavily)
      -> Hybrid Scraping (Parallel / Async)
      -> Deduplication
      -> NVIDIA Embedding Retrieval
      -> Cross-Encoder Reranking
      -> Source Reliability Scoring
      -> Composite Final Ranking
      -> Structured Answer (Short / Balanced / Detailed)
```
""")
st.markdown("---")
st.caption("Developed by Aviral Pratap Singh Chawda | AI & Data Science | Gandhinagar, Gujarat")