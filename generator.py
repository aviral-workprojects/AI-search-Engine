"""
Generator Module - LLM-based Answer Generation with Citations
Implements task-specific decoding parameters for optimal quality
"""

import logging
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# Task-specific decoding parameters (from research)
DECODING_CONFIGS = {
    'qa': {
        'temperature': 0.2,
        'top_p': 0.85,
        'max_tokens': 500
    },
    'summarization': {
        'temperature': 0.3,
        'top_p': 0.9,
        'max_tokens': 400
    },
    'general': {
        'temperature': 0.3,
        'top_p': 0.9,
        'max_tokens': 500
    }
}


@retry(stop=stop_after_attempt(3), 
       wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_answer_groq(query: str, context: str, groq_client, 
                        task_type: str = 'qa') -> str:
    """
    Generate answer using Groq with citations
    
    Args:
        query: User's question
        context: Article context with sources
        groq_client: Groq client instance
        task_type: 'qa', 'summarization', or 'general'
    
    Returns:
        Generated answer with citations
    """
    logger.info(f"Generating answer with Groq (task: {task_type})")
    
    # Get task-specific config
    config = DECODING_CONFIGS.get(task_type, DECODING_CONFIGS['general'])
    
    prompt = f"""You are a helpful AI assistant that answers questions based on provided sources.

CRITICAL RULES:
1. Answer ONLY using information from the provided sources
2. Cite sources using [1], [2], [3] format after each claim
3. Do NOT hallucinate or add information not in sources
4. If information is not in sources, say "The provided sources don't contain information about..."
5. Be direct and concise
6. Preserve all numbers, statistics, and dates exactly

USER QUESTION:
{query}

SOURCES:
{context}

ANSWER (with citations):"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_tokens=config['max_tokens']
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info("Groq answer generation successful")
        return answer
        
    except Exception as e:
        logger.error(f"Groq generation failed: {e}")
        raise


@retry(stop=stop_after_attempt(3), 
       wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_answer_gemini(query: str, context: str, gemini_model,
                          task_type: str = 'qa') -> str:
    """
    Generate answer using Gemini with citations
    
    Args:
        query: User's question
        context: Article context with sources
        gemini_model: Gemini model instance
        task_type: 'qa', 'summarization', or 'general'
    
    Returns:
        Generated answer with citations
    """
    logger.info(f"Generating answer with Gemini (task: {task_type})")
    
    # Get task-specific config
    config = DECODING_CONFIGS.get(task_type, DECODING_CONFIGS['general'])
    
    prompt = f"""You are a helpful AI assistant that answers questions based on provided sources.

CRITICAL RULES:
1. Answer ONLY using information from the provided sources
2. Cite sources using [1], [2], [3] format after each claim
3. Do NOT hallucinate or add information not in sources
4. If information is not in sources, say "The provided sources don't contain information about..."
5. Be direct and concise
6. Preserve all numbers, statistics, and dates exactly

USER QUESTION:
{query}

SOURCES:
{context}

ANSWER (with citations):"""

    try:
        # Configure generation
        generation_config = {
            'temperature': config['temperature'],
            'top_p': config['top_p'],
            'max_output_tokens': config['max_tokens'],
        }
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        answer = response.text.strip()
        logger.info("Gemini answer generation successful")
        return answer
        
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        raise


def generate_answer(query: str, context: str, llm_choice: str,
                   groq_client, gemini_model, 
                   task_type: str = 'qa') -> str:
    """
    Generate answer with automatic fallback
    
    Args:
        query: User's question
        context: Article context
        llm_choice: 'Groq' or 'Gemini'
        groq_client: Groq client
        gemini_model: Gemini model
        task_type: Task type for decoding config
    
    Returns:
        Generated answer with citations
    """
    if llm_choice == "Groq" and groq_client:
        try:
            return generate_answer_groq(query, context, groq_client, task_type)
        except Exception as e:
            logger.warning(f"Groq failed, trying Gemini: {e}")
            if gemini_model:
                return generate_answer_gemini(query, context, gemini_model, task_type)
            else:
                return "Error: No LLM available for answer generation."
    
    elif llm_choice == "Gemini" and gemini_model:
        try:
            return generate_answer_gemini(query, context, gemini_model, task_type)
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")
            return "Error: Answer generation failed. Please try again."
    
    else:
        return "Error: No LLM configured. Please set up Groq or Gemini API keys."


def extract_citations(answer: str) -> list:
    """
    Extract citation numbers from answer
    
    Args:
        answer: Generated answer text
    
    Returns:
        List of citation numbers found
    """
    import re
    
    # Find all [1], [2], etc.
    citations = re.findall(r'\[(\d+)\]', answer)
    
    # Convert to int and deduplicate
    citations = sorted(set(int(c) for c in citations))
    
    return citations