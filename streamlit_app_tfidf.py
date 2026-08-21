import os
import time

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
        "GROQ_API_KEY environment variable not found. "
        "Please add it in Render Environment Variables."
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
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
    "qwen/qwen3.6-27b": "Qwen 3.6 27B"
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

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f"Documents folder not found: {folder}"
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

                    chunks.append(chunk)

                    sources.append(filename)

    if not chunks:

        raise ValueError(
            "No valid .txt documents found in _data folder."
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
                "You are a helpful RAG assistant. "
                "Answer only using the provided context. "
                "If the answer is not available in the context, "
                "say: 'The answer is not available in the provided documents.' "
                "\n\n"
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
    "📚 MULTI-MODEL ANSWERBOARD "
)

st.write(
    "Cosine Similarity for document retrieval."
)


with st.sidebar:

    selected_models = st.multiselect(
        "Select Models",
        options=list(MODELS.keys()),
        default=[
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b"
        ],
        format_func=lambda x: MODELS[x]
    )

    top_k = st.slider(
        "Top K Chunks",
        min_value=1,
        max_value=5,
        value=3
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
            "Please enter a question."
        )

        st.stop()


    if not selected_models:

        st.warning(
            "Please select at least one model."
        )

        st.stop()


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
            "in the available documents."
        )

        st.stop()


    st.subheader(
        "📄 Retrieved Sources"
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

                answer, latency, error = call_model(
                    model,
                    question,
                    context
                )


            if error:

                st.error(
                    f"Model Error: {error}"
                )

            else:

                st.write(answer)

                st.metric(
                    "Latency",
                    f"{latency} ms"
                )
