"""Pydantic models = request/response contracts for the API."""
from typing import List
from pydantic import BaseModel, field_validator


class ExplainRequest(BaseModel):
    repo_url: str

    @field_validator("repo_url")
    @classmethod
    def must_be_github_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if v.endswith(".git"):
            v = v[:-4]
        if not v.startswith("https://github.com/") or len(v.split("/")) < 5:
            raise ValueError("Enter a URL like https://github.com/username/repository")
        return v


class ExplainResponse(BaseModel):
    repo_url: str
    model: str
    files_analyzed: List[str]
    explanation: str
