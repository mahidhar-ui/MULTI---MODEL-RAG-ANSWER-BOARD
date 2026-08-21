import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Multi-Model RAG Answerboard",
    page_icon="🤖",
    layout="wide"
)


@st.cache_data
def get_models():

    response = requests.get(
        f"{API_URL}/models",
        timeout=20
    )

    response.raise_for_status()

    return response.json()["models"]


try:

    MODELS = get_models()

except Exception as error:

    st.error(
        "Cannot connect to FastAPI backend.\n\n"
        "Run:\n"
        "python -m uvicorn main:app --reload --port 8000"
    )

    st.stop()


st.title(
    "🤖 Multi-Model RAG Answerboard"
)

st.write(
    "Ask one question and compare answers "
    "from multiple AI models."
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Settings")

    selected_models = st.multiselect(
        "Select Models",
        options=list(MODELS.keys()),
        default=list(MODELS.keys())[:2],
        format_func=lambda x:
        MODELS[x]["label"]
    )

    use_rag = st.toggle(
        "Enable RAG",
        value=True
    )

    smart_routing = st.toggle(
        "Enable Smart Routing",
        value=False
    )


# ==========================================
# QUESTION
# ==========================================

question = st.text_area(
    "Ask a Question",
    placeholder=(
        "Example: Why does RAG reduce hallucination?"
    ),
    height=150
)


if st.button(
    "Get Answers",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        payload = {
            "prompt": question,
            "models": selected_models,
            "use_rag": use_rag,
            "smart_routing": smart_routing
        }

        try:

            with st.spinner(
                "Generating answers..."
            ):

                response = requests.post(
                    f"{API_URL}/chat",
                    json=payload,
                    timeout=180
                )

                response.raise_for_status()

                data = response.json()


            # --------------------------------
            # ROUTING
            # --------------------------------

            if data.get("routing_info"):

                info = data["routing_info"]

                st.info(
                    f"Smart Routing Category: "
                    f"{info['category']}"
                )


            # --------------------------------
            # SOURCES
            # --------------------------------

            if data.get("sources"):

                st.subheader(
                    "📚 Sources Used"
                )

                for source in data["sources"]:

                    st.write(
                        f"📄 {source}"
                    )


            # --------------------------------
            # ANSWERS
            # --------------------------------

            results = data["results"]

            columns = st.columns(
                len(results)
            )

            for column, result in zip(
                columns,
                results
            ):

                with column:

                    st.subheader(
                        result["label"]
                    )

                    if result["error"]:

                        st.error(
                            result["error"]
                        )

                    else:

                        st.write(
                            result["answer"]
                        )

                        st.metric(
                            "Latency",
                            f"{result['latency_ms']} ms"
                        )

                        st.metric(
                            "Estimated Cost",
                            f"${result['estimated_cost_usd']:.6f}"
                        )

                        st.caption(
                            f"Input Tokens: "
                            f"{result['input_tokens']}"
                        )

                        st.caption(
                            f"Output Tokens: "
                            f"{result['output_tokens']}"
                        )


        except Exception as error:

            st.error(
                f"Error: {error}"
            )