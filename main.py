import os
import time
import asyncio
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import chromadb

from models_config import (
    MODELS,
    DEFAULT_MODELS,
    estimate_cost,
    approx_token_count
)

from router import route


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Add it to your .env file."
    )


# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(__name__)


# ==========================================
# GROQ CLIENT
# ==========================================

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Multi-Model RAG Answerboard"
)


# ==========================================
# CHROMADB
# ==========================================

chroma_client = chromadb.EphemeralClient()


def load_and_index_documents():

    data_folder = os.path.join(
        os.path.dirname(__file__),
        "_data"
    )

    os.makedirs(
        data_folder,
        exist_ok=True
    )

    try:
        chroma_client.delete_collection(
            name="rag_documents"
        )
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="rag_documents"
    )

    documents = []
    ids = []
    metadatas = []

    chunk_number = 0

    for filename in sorted(
        os.listdir(data_folder)
    ):

        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(
            data_folder,
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

            chunk_words = words[
                start:start + chunk_size
            ]

            chunk = " ".join(
                chunk_words
            ).strip()

            if len(chunk) < 30:
                continue

            documents.append(chunk)

            ids.append(
                f"chunk_{chunk_number}"
            )

            metadatas.append(
                {
                    "source": filename,
                    "chunk_number": chunk_number
                }
            )

            chunk_number += 1

    if documents:

        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

        logger.info(
            f"Indexed {len(documents)} chunks."
        )

    return collection


collection = load_and_index_documents()


# ==========================================
# RAG RETRIEVAL
# ==========================================

def retrieve_context(
    question: str,
    top_k: int = 3
):

    if collection.count() == 0:
        return "", []

    result_count = min(
        top_k,
        collection.count()
    )

    results = collection.query(
        query_texts=[question],
        n_results=result_count
    )

    chunks = results["documents"][0]

    metadatas = results["metadatas"][0]

    context = "\n\n".join(chunks)

    sources = []

    for metadata in metadatas:

        source = metadata["source"]

        if source not in sources:
            sources.append(source)

    return context, sources


# ==========================================
# CALL MODEL
# ==========================================

async def call_model(
    model: str,
    prompt: str,
    context: str = ""
):

    messages = []

    if context:

        messages.append(
            {
                "role": "system",
                "content": (
                    "Answer ONLY using the "
                    "provided context.\n\n"
                    "If the answer is not found "
                    "in the context, say:\n"
                    "'The answer is not available "
                    "in the provided documents.'\n\n"
                    f"CONTEXT:\n{context}"
                )
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    start_time = time.perf_counter()

    try:

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=700
        )

        latency_ms = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        usage = response.usage

        input_tokens = (
            usage.prompt_tokens
            if usage
            else approx_token_count(
                prompt + context
            )
        )

        output_tokens = (
            usage.completion_tokens
            if usage
            else approx_token_count(
                answer
            )
        )

        estimated_cost = estimate_cost(
            model,
            input_tokens,
            output_tokens
        )

        return {
            "model": model,
            "label": MODELS[model]["label"],
            "answer": answer,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost,
            "error": None
        }

    except Exception as error:

        return {
            "model": model,
            "label": MODELS[model]["label"],
            "answer": None,
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "error": str(error)
        }


# ==========================================
# REQUEST MODELS
# ==========================================

class ChatRequest(BaseModel):

    prompt: str

    models: list[str] = DEFAULT_MODELS

    use_rag: bool = False

    smart_routing: bool = False


class ABRequest(BaseModel):

    prompt_a: str

    prompt_b: str

    model: str = "llama-3.1-8b-instant"

    use_rag: bool = False


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Multi-Model RAG Answerboard is running."
    }


# ==========================================
# GET MODELS
# ==========================================

@app.get("/models")
def get_models():

    return {
        "models": MODELS
    }


# ==========================================
# GET DOCUMENT INFO
# ==========================================

@app.get("/documents")
def get_documents():

    data_folder = os.path.join(
        os.path.dirname(__file__),
        "_data"
    )

    files = []

    for filename in os.listdir(
        data_folder
    ):

        if filename.endswith(".txt"):
            files.append(filename)

    return {
        "documents": files,
        "total_chunks": collection.count()
    }


# ==========================================
# CHAT ENDPOINT
# ==========================================

@app.post("/chat")
async def chat(
    request: ChatRequest
):

    if not request.prompt.strip():

        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty."
        )

    context = ""
    sources = []

    if request.use_rag:

        context, sources = retrieve_context(
            request.prompt
        )

    selected_models = request.models

    routing_info = None

    if request.smart_routing:

        selected_models, category = route(
            request.prompt,
            MODELS
        )

        routing_info = {
            "category": category,
            "chosen_models": selected_models
        }

    invalid_models = [
        model
        for model in selected_models
        if model not in MODELS
    ]

    if invalid_models:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid models: {invalid_models}"
        )

    tasks = [

        call_model(
            model,
            request.prompt,
            context
        )

        for model in selected_models
    ]

    results = await asyncio.gather(
        *tasks
    )

    return {
        "results": results,
        "sources": sources,
        "routing_info": routing_info
    }


# ==========================================
# A/B TEST ENDPOINT
# ==========================================

@app.post("/ab_test")
async def ab_test(
    request: ABRequest
):

    if request.model not in MODELS:

        raise HTTPException(
            status_code=400,
            detail="Invalid model."
        )

    context_a = ""
    context_b = ""

    if request.use_rag:

        context_a, _ = retrieve_context(
            request.prompt_a
        )

        context_b, _ = retrieve_context(
            request.prompt_b
        )

    result_a, result_b = await asyncio.gather(

        call_model(
            request.model,
            request.prompt_a,
            context_a
        ),

        call_model(
            request.model,
            request.prompt_b,
            context_b
        )

    )

    return {
        "variant_a": result_a,
        "variant_b": result_b
    }