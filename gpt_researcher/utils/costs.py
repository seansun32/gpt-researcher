"""Cost estimation utilities for LLM API usage.

This module provides functions to estimate the cost of LLM API calls
based on token counts. Uses a simple character-based approximation
to avoid external network calls (tiktoken downloads encoding files
from openaipublic.blob.core.windows.net at runtime).
"""

import logging

logger = logging.getLogger(__name__)

INPUT_COST_PER_TOKEN = 0.000005
OUTPUT_COST_PER_TOKEN = 0.000015
IMAGE_INFERENCE_COST = 0.003825
EMBEDDING_COST = 0.02 / 1000000  # Assumes new ada-3-small


def _approx_token_count(text: str) -> int:
    """Approximate token count without tiktoken.

    Uses a simple heuristic: ~1 token per 4 characters for English,
    ~1 token per 1.5 characters for CJK-heavy text.
    """
    if not text:
        return 0
    # Count CJK characters (Chinese, Japanese, Korean)
    cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff'
                    or '\u3040' <= ch <= '\u30ff'
                    or '\uac00' <= ch <= '\ud7af')
    cjk_ratio = cjk_count / len(text) if text else 0
    if cjk_ratio > 0.3:
        # CJK-heavy text: ~1.5 chars per token
        return max(1, int(len(text) / 1.5))
    else:
        # English/Latin text: ~4 chars per token
        return max(1, len(text) // 4)


def estimate_llm_cost(input_content: str, output_content: str) -> float:
    """Estimate the cost of an LLM API call based on input and output content.

    Cost estimation is based on OpenAI pricing and may vary for other models.

    Args:
        input_content: The input text sent to the LLM.
        output_content: The output text received from the LLM.

    Returns:
        The estimated cost in USD.
    """
    input_token_count = _approx_token_count(input_content)
    output_token_count = _approx_token_count(output_content)
    input_costs = input_token_count * INPUT_COST_PER_TOKEN
    output_costs = output_token_count * OUTPUT_COST_PER_TOKEN
    return input_costs + output_costs


def estimate_embedding_cost(model: str, docs: list) -> float:
    """Estimate the cost of embedding documents.

    Args:
        model: The embedding model name.
        docs: List of documents to embed.

    Returns:
        The estimated embedding cost in USD.
    """
    total_tokens = sum(_approx_token_count(str(doc)) for doc in docs)
    return total_tokens * EMBEDDING_COST
