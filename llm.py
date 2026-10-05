"""Talks to the LOCAL LLM. Two interchangeable backends: Ollama or Hugging Face Transformers."""
import requests

from . import config

SYSTEM_PROMPT = (
    "You are a helpful senior developer who explains code to beginners in simple, "
    "plain English. Only describe what is actually visible in the code provided. "
    "Do not invent features."
)

USER_TEMPLATE = """Below are the file tree and key source files of a GitHub repository.

{context}

Write an explanation using EXACTLY this format:

## Project Overview
(2-3 simple sentences: what is this project?)

## What the application allows users to do
(bullet list)

## Main Technologies
(bullet list of languages, frameworks, databases, libraries)

## How it works
(short step-by-step description of how the parts talk to each other)

## Important Files
(bullet list: file name - what it does)
"""


def _build_messages(context: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context)},
    ]


# ---------------------------------------------------------------- Ollama ----
def _ollama_explain(context: str) -> str:
    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": _build_messages(context),
                "stream": False,
                "options": {"num_ctx": config.NUM_CTX, "temperature": 0.3},
            },
            timeout=600,
        )
    except requests.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama. Start it (open the Ollama app or run `ollama serve`) "
            f"and make sure the model is pulled: `ollama pull {config.OLLAMA_MODEL}`"
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
    return resp.json()["message"]["content"]


# ----------------------------------------------------- Hugging Face local ----
_hf_cache = {}


def _hf_explain(context: str) -> str:
    # Imported lazily so Ollama users don't need torch/transformers installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "model" not in _hf_cache:
        tok = AutoTokenizer.from_pretrained(config.HF_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            config.HF_MODEL, torch_dtype="auto", device_map="auto"
        )
        _hf_cache["tok"], _hf_cache["model"] = tok, model
    tok, model = _hf_cache["tok"], _hf_cache["model"]

    prompt = tok.apply_chat_template(
        _build_messages(context), tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=600, do_sample=True, temperature=0.3)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


# ------------------------------------------------------------- public API ---
def model_name() -> str:
    return config.OLLAMA_MODEL if config.LLM_BACKEND == "ollama" else config.HF_MODEL


def explain(context: str) -> str:
    if config.LLM_BACKEND == "hf":
        return _hf_explain(context)
    return _ollama_explain(context)
