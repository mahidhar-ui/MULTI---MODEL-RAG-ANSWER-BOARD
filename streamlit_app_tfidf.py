import os
import time

import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)


st.set_page_config(
    page_title="TF-IDF RAG",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# API KEY
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY not found in .env"
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
    "llama-3.1-8b-instant":
    "Llama 3.1 8B",

    "llama-3.3-70b-versatile":
    "Llama 3.3 70B",

    "openai/gpt-oss-20b":
    "GPT-OSS 20B",

    "openai/gpt-oss-120b":
    "GPT-OSS 120B"
}


# ==========================================
# LOAD DOCUMENTS
# ==========================================

@st.cache_resource
def load_documents():

    folder = os.path.join(
        os.path.dirname(__file__),
        "_data"
    )

    chunks = []

    sources = []

    for filename in os.listdir(folder):

        if filename.endswith(".txt"):

            filepath = os.path.join(
                folder,
                filename
            )

            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

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

                if len(chunk) > 30:

                    chunks.append(
                        chunk
                    )

                    sources.append(
                        filename
                    )

    return chunks, sources


# ==========================================
# TF-IDF INDEX
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
# CALL MODEL
# ==========================================

def call_model(
    model,
    question,
    context
):

    messages = [
        {
            "role": "system",
            "content": (
                "Answer only using this context. "
                "If the answer is not available, "
                "say so.\n\n"
                f"CONTEXT:\n{context}"
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    start = time.perf_counter()

    try:

        response = (
            client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=700
            )
        )

        latency = round(
            (
                time.perf_counter()
                - start
            ) * 1000
        )

        return (
            response
            .choices[0]
            .message
            .content,
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
    "📚 TF-IDF RAG Answerboard"
)

st.write(
    "This version uses TF-IDF + "
    "Cosine Similarity for retrieval."
)


with st.sidebar:

    selected_models = st.multiselect(
        "Select Models",
        options=list(MODELS.keys()),
        default=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile"
        ],
        format_func=lambda x:
        MODELS[x]
    )

    top_k = st.slider(
        "Top K Chunks",
        1,
        5,
        3
    )


question = st.text_area(
    "Ask a question",
    height=150
)


if st.button(
    "Search and Answer",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Enter a question."
        )

        st.stop()


    context, sources = retrieve(
        question,
        top_k
    )


    st.subheader(
        "Retrieved Sources"
    )

    for source in sources:

        st.write(
            f"📄 {source}"
        )


    columns = st.columns(
        len(selected_models)
    )


    for column, model in zip(
        columns,
        selected_models
    ):

        with column:

            st.subheader(
                MODELS[model]
            )

            with st.spinner(
                "Generating..."
            ):

                answer, latency, error = (
                    call_model(
                        model,
                        question,
                        context
                    )
                )

            if error:

                st.error(error)

            else:

                st.write(answer)

                st.metric(
                    "Latency",
                    f"{latency} ms"
                )