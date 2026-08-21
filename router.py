"""
router.py
---------
Smart routing for the Multi-Model RAG Answerboard.

The router analyzes the user's prompt and chooses
appropriate models.
"""

import re


CODE_HINTS = re.compile(
    r"\b("
    r"code|function|bug|error|python|java|sql|"
    r"regex|algorithm|debug|api|class|script|"
    r"program|coding"
    r")\b",
    re.IGNORECASE,
)


REASONING_HINTS = re.compile(
    r"\b("
    r"why|explain|analyze|analyse|compare|"
    r"trade-?off|design|architecture|strategy|"
    r"prove|reason|evaluate"
    r")\b",
    re.IGNORECASE,
)


SHORT_FACTUAL_HINTS = re.compile(
    r"^(what|who|when|where|is|are|does|do|define|list)\b",
    re.IGNORECASE,
)


def classify_prompt(prompt: str) -> str:
    """
    Classify a prompt into one of the routing categories.
    """

    prompt = prompt.strip()

    word_count = len(prompt.split())

    if CODE_HINTS.search(prompt):
        return "code"

    if REASONING_HINTS.search(prompt):
        return "reasoning"

    if word_count > 40:
        return "long_form"

    if (
        SHORT_FACTUAL_HINTS.match(prompt)
        and word_count <= 15
    ):
        return "short_factual"

    return "simple_qa"


def route(prompt: str, models: dict):
    """
    Select suitable models for a prompt.

    Returns:
        chosen_models: list of model IDs
        category: detected prompt category
    """

    category = classify_prompt(prompt)

    candidates = []

    for model_id, config in models.items():
        if category in config["good_for"]:
            candidates.append(model_id)

    if not candidates:
        candidates = list(models.keys())

    candidates.sort(
        key=lambda model_id: models[
            model_id
        ]["input_price_per_m"]
    )

    chosen = candidates[:1]

    flagship_models = [
        model_id
        for model_id, config in models.items()
        if config["tier"] == "flagship"
        and model_id not in chosen
    ]

    if flagship_models:
        chosen.append(flagship_models[0])

    return chosen, category