import os

HF_TOKEN = os.getenv("HF_TOKEN")

HUGGINGFACE_REPO_ID = "meta-llama/Llama-2-7b-chat-hf"

DB_FAISS_PATH = "vectorstore/db_faiss"

DATA_PATH = "data/"

CHUNK_SIZE = 200

CHUNK_OVERLAP = 50

groq_api_key = os.getenv("GROQ_API_KEY")


