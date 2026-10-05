"""Central configuration. Override any value with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.getenv("WORKSPACE", BASE_DIR / "workspace"))  # where repos get cloned

# --- LLM settings -----------------------------------------------------------
# "ollama" (default) or "hf" (Hugging Face Transformers, runs fully in Python)
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# --- Repo processing limits (small local models have small context windows) --
MAX_FILE_CHARS = int(os.getenv("MAX_FILE_CHARS", 2500))     # per file
MAX_TOTAL_CHARS = int(os.getenv("MAX_TOTAL_CHARS", 9000))   # all code sent to the LLM
MAX_FILE_SIZE_BYTES = 200_000                                # skip files bigger than this
MAX_FILES = int(os.getenv("MAX_FILES", 12))

# Ollama context window in tokens (must be >= prompt size; ~4 chars per token)
NUM_CTX = int(os.getenv("NUM_CTX", 8192))
