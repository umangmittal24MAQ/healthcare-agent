from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Queue
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError

from app.config import get_llm_config
from app.llm import LLMError, probe_llm
from app.pipeline import PipelineStageError, run_analysis
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


@app.get("/health/llm")
def health_llm():
    try:
        result = probe_llm()
        return {"status": "ok", **result}
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/analyze/stream")
def analyze_stream(request: AnalyzeRequest):
    def event_stream():
        queue: Queue = Queue()
        sentinel = object()

        def emit(event: dict):
            queue.put(event)

        def worker():
            try:
                result = run_analysis(request, emit=emit)
                emit(
                    {
                        "type": "final",
                        "result": result.model_dump(mode="json"),
                    }
                )
            except PipelineStageError as exc:
                emit(
                    {
                        "type": "error",
                        "stage": exc.stage,
                        "message": str(exc),
                    }
                )
            except (LLMError, ValidationError, ValueError) as exc:
                emit(
                    {
                        "type": "error",
                        "stage": "unknown",
                        "message": str(exc),
                    }
                )
            except Exception as exc:
                emit(
                    {
                        "type": "error",
                        "stage": "server",
                        "message": str(exc),
                    }
                )
            finally:
                queue.put(sentinel)

        Thread(target=worker, daemon=True).start()

        while True:
            event = queue.get()
            if event is sentinel:
                break
            yield "data: " + json.dumps(event, ensure_ascii=False, default=str) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/analyze", response_model=AnalysisResult)
def analyze(request: AnalyzeRequest):
    try:
        return run_analysis(request)
    except PipelineStageError as exc:
        raise HTTPException(
            status_code=502,
            detail={"stage": exc.stage, "message": str(exc)},
        ) from exc
    except (LLMError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=502, detail={"stage": "unknown", "message": str(exc)}) from exc


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
