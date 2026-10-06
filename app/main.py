from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError

from app.config import get_llm_config
from app.llm import LLMError
from app.pipeline import run_analysis
from app.schemas import AnalysisResult, AnalyzeRequest, ReviewRequest
from app.store import get_run, init_db, save_review

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Diagnostic Decision Support Agent",
    version="2.0.0",
    description="Synthetic-data clinical decision support with retrieve-first reasoning and a blind challenge layer.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    model = get_llm_config()
    return {"status": "ok", "model": model.id, "llm_endpoint": str(model.url)}


@app.post("/api/analyze", response_model=AnalysisResult)
def analyze(request: AnalyzeRequest):
    try:
        return run_analysis(request)
    except (LLMError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}")
def read_run(run_id: str):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/api/runs/{run_id}/review")
def review_run(run_id: str, review: ReviewRequest):
    try:
        save_review(run_id, review.decision, review.comment, datetime.now(timezone.utc).isoformat())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
    return {"run_id": run_id, "review": review.model_dump(), "status": "recorded"}
