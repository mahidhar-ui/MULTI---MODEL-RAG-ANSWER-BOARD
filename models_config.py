"""
models_config.py
----------------
Central configuration for all models used in the
Multi-Model RAG Answerboard.

These model IDs are sent to the Groq API.
"""

MODELS = {
    "openai/gpt-oss-20b": {
        "label": "GPT-OSS 20B",
        "tier": "fast",
        "input_price_per_m": 0.075,
        "output_price_per_m": 0.30,
        "good_for": [
            "short_factual",
            "simple_qa",
            "code",
            "chit_chat",
        ],
    },

    "openai/gpt-oss-120b": {
        "label": "GPT-OSS 120B",
        "tier": "flagship",
        "input_price_per_m": 0.15,
        "output_price_per_m": 0.60,
        "good_for": [
            "reasoning",
            "long_form",
            "code",
        ],
    },

    "qwen/qwen3.6-27b": {
        "label": "Qwen 3.6 27B",
        "tier": "mid",
        "input_price_per_m": 0.60,
        "output_price_per_m": 3.00,
        "good_for": [
            "reasoning",
            "long_form",
            "code",
        ],
    },
}


DEFAULT_MODELS = list(MODELS.keys())


def estimate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate the API cost of one model request.
    """

    cfg = MODELS[model_id]

    input_cost = (
        input_tokens / 1_000_000
    ) * cfg["input_price_per_m"]

    output_cost = (
        output_tokens / 1_000_000
    ) * cfg["output_price_per_m"]

    total_cost = input_cost + output_cost

    return round(total_cost, 6)


def approx_token_count(text: str) -> int:
    """
    Rough approximation of token count.

    This is used only for displaying estimated cost.
    """

    words = len(text.split())

    return max(1, int(words * 1.3))