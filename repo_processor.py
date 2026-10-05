"""GitHub repo -> clone -> pick relevant files -> build a text 'context' for the LLM."""
import shutil
import stat
from pathlib import Path
from typing import List, Tuple

from git import Repo  # GitPython

from . import config

# Folders that never contain useful source code.
IGNORE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__", "dist", "build",
    ".idea", ".vscode", "target", "vendor", ".next", "coverage", "site-packages",
    "migrations", "static", "assets", "images", "tests", "test", "examples", "docs",
}

# Extensions we treat as source code / useful config.
CODE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".cs", ".go",
    ".rs", ".php", ".rb", ".kt", ".swift", ".html", ".css", ".sql", ".sh", ".r",
}

# Files that explain a project the fastest -> read these first.
PRIORITY_NAMES = [
    "readme.md", "readme.rst", "readme.txt", "readme",
    "requirements.txt", "package.json", "pom.xml", "build.gradle", "pyproject.toml",
    "composer.json", "go.mod", "cargo.toml", "dockerfile",
    "main.py", "app.py", "server.py", "index.js", "app.js", "server.js", "main.java",
    "index.html", "manage.py",
]


def _force_remove(func, path, _exc):
    """Windows keeps .git files read-only; clear the flag so rmtree can delete them."""
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def clone_repo(repo_url: str) -> Path:
    """Shallow-clone (latest commit only) into workspace/<owner>__<repo>."""
    owner, name = repo_url.split("/")[-2:]
    dest = config.WORKSPACE / f"{owner}__{name}"
    config.WORKSPACE.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest, onerror=_force_remove)
    Repo.clone_from(repo_url, dest, depth=1)
    return dest


def _is_wanted(rel: Path, root: Path) -> bool:
    if any(part in IGNORE_DIRS for part in rel.parts):
        return False
    if (root / rel).stat().st_size > config.MAX_FILE_SIZE_BYTES:
        return False
    # Special files (README, package.json...) only count near the repo root,
    # otherwise dozens of nested READMEs/configs crowd out the real source code.
    is_priority = rel.name.lower() in PRIORITY_NAMES and len(rel.parts) <= 2
    return rel.suffix.lower() in CODE_EXTS or is_priority


def _rank(path: Path, root: Path) -> Tuple[int, int, str]:
    """Lower = more important. Priority files first, then shallow files, then A-Z."""
    name = path.name.lower()
    depth = len(path.relative_to(root).parts)
    prio = PRIORITY_NAMES.index(name) if (name in PRIORITY_NAMES and depth <= 2) else len(PRIORITY_NAMES)
    return (prio, depth, str(path))


def select_files(root: Path) -> List[Path]:
    candidates = [
        p for p in root.rglob("*")
        if p.is_file() and _is_wanted(p.relative_to(root), root)
    ]
    candidates.sort(key=lambda p: _rank(p, root))
    return candidates[: config.MAX_FILES]


def file_tree(root: Path, limit: int = 60) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        if p.is_file():
            lines.append(str(rel))
        if len(lines) >= limit:
            lines.append("... (truncated)")
            break
    return "\n".join(lines)


def build_context(root: Path) -> Tuple[str, List[str]]:
    """Return (text for the LLM, list of file names actually included)."""
    parts = [f"FILE TREE:\n{file_tree(root)}\n"]
    used: List[str] = []
    budget = config.MAX_TOTAL_CHARS

    for path in select_files(root):
        if budget <= 0:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        text = text.strip()
        if not text:
            continue
        snippet = text[: min(config.MAX_FILE_CHARS, budget)]
        rel = str(path.relative_to(root))
        parts.append(f"--- FILE: {rel} ---\n{snippet}\n")
        used.append(rel)
        budget -= len(snippet)

    return "\n".join(parts), used
