"""FastAPI backend. Run:  uvicorn app.main:app --reload --port 8000"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from git.exc import GitCommandError

from . import config, llm, repo_processor
from .schemas import ExplainRequest, ExplainResponse

app = FastAPI(title="Local GitHub Repository Code Explainer")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok", "backend": config.LLM_BACKEND, "model": llm.model_name()}


@app.post("/explain", response_model=ExplainResponse)
def explain_repo(req: ExplainRequest):
    # 1. Clone
    try:
        root = repo_processor.clone_repo(req.repo_url)
    except GitCommandError as e:
        raise HTTPException(400, f"Could not clone repository (is it public?): {e}")

    # 2-3. Identify relevant files + extract code
    context, files = repo_processor.build_context(root)
    if not files:
        raise HTTPException(422, "No readable source-code files found in this repository.")

    # 4-5. Local LLM generates the explanation
    try:
        explanation = llm.explain(context)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return ExplainResponse(
        repo_url=req.repo_url,
        model=llm.model_name(),
        files_analyzed=files,
        explanation=explanation,
    )
