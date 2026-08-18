"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# OpenAI is the assignment's default. Gemini can be selected explicitly and is
# called through Google's OpenAI-compatible endpoint, so no extra SDK is needed.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EVAL_CHAT_MODEL = os.getenv("OPENAI_EVAL_CHAT_MODEL", OPENAI_CHAT_MODEL)
OPENAI_EVAL_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EVAL_EMBEDDING_MODEL", "text-embedding-3-small"
)
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
GEMINI_EVAL_CHAT_MODEL = os.getenv(
    "GEMINI_EVAL_CHAT_MODEL", "gemini-3.1-flash-lite"
)
GEMINI_EVAL_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EVAL_EMBEDDING_MODEL", "gemini-embedding-001"
)
GEMINI_OPENAI_BASE_URL = os.getenv(
    "GEMINI_OPENAI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
# 0 selects a provider-aware default (OpenAI: 4, Gemini: 1).
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "0"))
# Conservative defaults for Gemini's free tier (15 generate-content RPM).
# Keeping a little headroom prevents RAGAS/enrichment jobs from becoming
# zero-score fallbacks when the provider responds with HTTP 429.
GEMINI_RAGAS_RPM = float(os.getenv("GEMINI_RAGAS_RPM", "12"))
GEMINI_ENRICH_RPM = float(os.getenv("GEMINI_ENRICH_RPM", "12"))

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
