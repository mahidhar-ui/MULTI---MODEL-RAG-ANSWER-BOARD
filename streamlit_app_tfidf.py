import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MULTI-MODEL RAG ANSWERBOARD",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# API KEY
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY environment variable not found."
    )

    st.stop()


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# ==========================================
# MODELS
# ==========================================

MODELS = {

    "openai/gpt-oss-20b": {
        "name": "GPT-OSS 20B",
        "cost": "Low",
        "speed": "Very Fast"
    },

    "openai/gpt-oss-120b": {
        "name": "GPT-OSS 120B",
        "cost": "Medium",
        "speed": "Fast"
    },

    "qwen/qwen3.6-27b": {
        "name": "Qwen 3.6 27B",
        "cost": "High",
        "speed": "Fast"
    }
}


# ==========================================
# PROMPTS
# ==========================================

PROMPTS = {

    "Prompt A - Concise": (
        "Answer clearly and concisely. "
        "Focus only on the most important information."
    ),

    "Prompt B - Detailed": (
        "Act as an expert teacher. "
        "Explain the answer step by step "
        "using clear and simple language."
    )
}


# ==========================================
# LOAD DOCUMENTS
# ==========================================

@st.cache_resource
def load_documents():

    base_folder = Path(__file__).resolve().parent

    # Check all possible document folders
    possible_folders = [
        base_folder / "_data",
        base_folder / "DOCS",
        base_folder / "docs"
    ]

    chunks = []
    sources = []
    found_files = []

    # Go through every possible folder
    for folder in possible_folders:

        if folder.exists() and folder.is_dir():

            txt_files = list(
                folder.glob("*.txt")
            )

            for filepath in txt_files:

                try:

                    with open(
                        filepath,
                        "r",
                        encoding="utf-8"
                    ) as file:

                        text = file.read()

                    # Skip empty files
                    if not text.strip():
                        continue

                    found_files.append(
                        str(filepath)
                    )

                    words = text.split()

                    chunk_size = 80

                    for start in range(
                        0,
                        len(words),
                        chunk_size
                    ):

                        chunk = " ".join(
                            words[
                                start:
                                start + chunk_size
                            ]
                        )

                        if len(chunk.strip()) > 30:

                            chunks.append(
                                chunk
                            )

                            sources.append(
                                filepath.name
                            )

                except Exception as error:

                    print(
                        f"Error reading {filepath}: {error}"
                    )


    # If no TXT files were found
    if not found_files:

        searched_folders = "\n".join(
            [
                str(folder)
                for folder in possible_folders
            ]
        )

        raise ValueError(
            "No .txt documents found.\n\n"
            "The application searched these folders:\n"
            f"{searched_folders}"
        )


    # If files exist but no valid chunks
    if not chunks:

        raise ValueError(
            "TXT files were found, but no valid "
            "text chunks could be created."
        )


    return chunks, sources


# ==========================================
# BUILD TF-IDF INDEX
# ==========================================

@st.cache_resource
def build_index():

    chunks, sources = load_documents()

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        chunks
    )

    return (
        vectorizer,
        matrix,
        chunks,
        sources
    )


# ==========================================
# RETRIEVAL
# ==========================================

def retrieve(
    question,
    top_k=3
):

    (
        vectorizer,
        matrix,
        chunks,
        sources
    ) = build_index()


    question_vector = vectorizer.transform(
        [question]
    )


    scores = cosine_similarity(
        question_vector,
        matrix
    ).flatten()


    indices = scores.argsort()[
        ::-1
    ][:top_k]


    relevant_chunks = []
    relevant_sources = []


    for index in indices:

        if scores[index] > 0:

            relevant_chunks.append(
                chunks[index]
            )

            relevant_sources.append(
                sources[index]
            )


    context = "\n\n".join(
        relevant_chunks
    )


    unique_sources = list(
        dict.fromkeys(
            relevant_sources
        )
    )


    return context, unique_sources


# ==========================================
# SMART ROUTING
# ==========================================

def smart_route(question):

    word_count = len(
        question.split()
    )

    question_lower = question.lower()


    complex_keywords = [
        "explain",
        "compare",
        "difference",
        "analyze",
        "analysis",
        "architecture",
        "algorithm",
        "design"
    ]


    coding_keywords = [
        "python",
        "code",
        "program",
        "function",
        "bug",
        "error",
        "machine learning"
    ]


    if any(
        keyword in question_lower
        for keyword in coding_keywords
    ):

        return "qwen/qwen3.6-27b"


    elif (
        word_count > 20
        or any(
            keyword in question_lower
            for keyword in complex_keywords
        )
    ):

        return "openai/gpt-oss-120b"


    else:

        return "openai/gpt-oss-20b"


# ==========================================
# CALL MODEL
# ==========================================

def call_model(
    model,
    question,
    context,
    rag_enabled,
    prompt_instruction
):

    if rag_enabled:

        system_message = (
            f"{prompt_instruction}\n\n"
            "Answer only using the provided context. "
            "If the answer is not available in the context, "
            "say: 'The answer is not available in the provided documents.'"
            "\n\n"
            f"CONTEXT:\n{context}"
        )

    else:

        system_message = (
            f"{prompt_instruction}\n\n"
            "Answer the user's question using your general knowledge."
        )


    messages = [

        {
            "role": "system",
            "content": system_message
        },

        {
            "role": "user",
            "content": question
        }

    ]


    start = time.perf_counter()


    try:

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=700
        )


        latency = round(
            (
                time.perf_counter()
                - start
            ) * 1000
        )


        return (
            response.choices[0].message.content,
            latency,
            None
        )


    except Exception as error:

        return (
            None,
            0,
            str(error)
        )


# ==========================================
# UI
# ==========================================

st.title(
    "🤖 Multi-Model RAG Answerboard"
)

st.write(
    "Ask one question and compare multiple AI models "
    "side-by-side with RAG, smart routing, "
    "and A/B prompt testing."
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Settings")


    rag_enabled = st.toggle(
        "Enable RAG",
        value=True
    )


    smart_routing = st.toggle(
        "Enable Smart Routing",
        value=False
    )


    ab_testing = st.toggle(
        "Enable A/B Prompt Testing",
        value=False
    )


    selected_models = st.multiselect(
        "Select Models",
        options=list(MODELS.keys()),
        default=[
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b"
        ],
        format_func=lambda x: MODELS[x]["name"]
    )


    top_k = st.slider(
        "Top K Chunks",
        min_value=1,
        max_value=5,
        value=3
    )


    st.divider()


    if smart_routing:

        st.info(
            "🧠 Smart Routing is enabled. "
            "The system will automatically select a model."
        )

    else:

        st.caption(
            "💡 Select one or more models "
            "to compare their answers."
        )


# ==========================================
# QUESTION INPUT
# ==========================================

question = st.text_area(
    "Ask a question",
    height=150,
    placeholder=(
        "Example: "
        "What is Retrieval-Augmented Generation?"
    )
)


# ==========================================
# SEARCH AND ANSWER
# ==========================================

if st.button(
    "Search and Answer",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # ======================================
    # SMART ROUTING
    # ======================================

    if smart_routing:

        routed_model = smart_route(
            question
        )

        models_to_use = [
            routed_model
        ]

        st.info(
            "🧠 Smart Router selected: "
            f"{MODELS[routed_model]['name']}"
        )

    else:

        models_to_use = selected_models


    if not models_to_use:

        st.warning(
            "Please select at least one model."
        )

        st.stop()


    # ======================================
    # RAG RETRIEVAL
    # ======================================

    context = ""
    sources = []


    if rag_enabled:

        try:

            context, sources = retrieve(
                question,
                top_k
            )

        except Exception as error:

            st.error(
                f"Error loading documents: {error}"
            )

            st.stop()


        if not context:

            st.warning(
                "No relevant information was found "
                "in the documents."
            )

            st.stop()


        st.subheader(
            "📄 Retrieved Sources"
        )


        for source in sources:

            st.write(
                f"📄 {source}"
            )


    # ======================================
    # A/B PROMPT TESTING
    # ======================================

    if ab_testing:

        st.subheader(
            "🧪 A/B Prompt Comparison"
        )


        for model in models_to_use:

            st.divider()

            st.subheader(
                MODELS[model]["name"]
            )


            col_a, col_b = st.columns(2)


            with col_a:

                st.markdown(
                    "### 🅰️ Prompt A - Concise"
                )

                with st.spinner(
                    "Generating Prompt A..."
                ):

                    answer, latency, error = (
                        call_model(
                            model,
                            question,
                            context,
                            rag_enabled,
                            PROMPTS[
                                "Prompt A - Concise"
                            ]
                        )
                    )


                if error:

                    st.error(error)

                else:

                    st.success(
                        f"⚡ {latency} ms"
                    )

                    st.write(answer)


            with col_b:

                st.markdown(
                    "### 🅱️ Prompt B - Detailed"
                )

                with st.spinner(
                    "Generating Prompt B..."
                ):

                    answer, latency, error = (
                        call_model(
                            model,
                            question,
                            context,
                            rag_enabled,
                            PROMPTS[
                                "Prompt B - Detailed"
                            ]
                        )
                    )


                if error:

                    st.error(error)

                else:

                    st.success(
                        f"⚡ {latency} ms"
                    )

                    st.write(answer)


    # ======================================
    # NORMAL MULTI-MODEL MODE
    # ======================================

    else:

        columns = st.columns(
            len(models_to_use)
        )


        for column, model in zip(
            columns,
            models_to_use
        ):

            with column:

                st.subheader(
                    MODELS[model]["name"]
                )


                badge_col1, badge_col2 = st.columns(2)


                with badge_col1:

                    st.caption(
                        f"💰 Cost: "
                        f"{MODELS[model]['cost']}"
                    )


                with badge_col2:

                    st.caption(
                        f"⚡ Expected: "
                        f"{MODELS[model]['speed']}"
                    )


                with st.spinner(
                    "Generating..."
                ):

                    answer, latency, error = (
                        call_model(
                            model,
                            question,
                            context,
                            rag_enabled,
                            PROMPTS[
                                "Prompt A - Concise"
                            ]
                        )
                    )


                if error:

                    st.error(
                        f"Model Error: {error}"
                    )

                else:

                    st.write(
                        answer
                    )


                    st.metric(
                        "Actual Latency",
                        f"{latency} ms"
                    )


                    if rag_enabled and sources:

                        st.markdown(
                            "#### 📚 Sources"
                        )


                        for source in sources:

                            st.caption(
                                f"📄 {source}"
                            )
