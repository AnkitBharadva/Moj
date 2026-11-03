"""
FastAPI service for comment classification.

Supports:
- Model: Qwen3.5-9B (qwen3.5-judge) and DeepSeek-V4-Pro (comment-judge)
- Built-in tasks: sentiment (3-class), sentiment_fine (5-class), hate, support, intent
- Few-shot Hinglish exemplars & clear boundary definitions
- Optional Cyclic Option-Order Debiasing (eliminating position bias)
- Dynamic targets: analyze stance towards any entity/campaign
- Dynamic custom tasks: pass arbitrary questions and options on the fly
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .readout import TASKS, CommentClassifier, TaskConfig, load_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_DIR = Path("config")


def load_calibration_config():
    """Load calibration temperatures and decision thresholds."""
    temps = {}
    threshs = {}

    t_file = CONFIG_DIR / "temperatures.json"
    if t_file.exists():
        try:
            raw_t = json.loads(t_file.read_text(encoding="utf-8"))
            temps = {k: v.get("T", 1.0) for k, v in raw_t.items()}
        except Exception as e:
            logger.warning("Could not read temperatures.json: %s", e)

    th_file = CONFIG_DIR / "thresholds.json"
    if th_file.exists():
        try:
            threshs = json.loads(th_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not read thresholds.json: %s", e)

    return temps, threshs


# Global classifier instances
classifier = CommentClassifier(model="qwen3.5-judge")
temperatures, thresholds = load_calibration_config()
request_history: List[Dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm model into GPU VRAM on startup."""
    global temperatures, thresholds
    temperatures, thresholds = load_calibration_config()
    logger.info("Pre-warming Ollama model (%s) into GPU VRAM...", classifier.model)
    try:
        classifier.analyze("Warmup query", task_names=["sentiment"])
        logger.info("Warmup complete. %s is hot in VRAM.", classifier.model)
    except Exception as e:
        logger.warning("Startup pre-warm failed: %s", e)
    yield


app = FastAPI(
    title="High-Accuracy Comment Classifier API",
    description="24/7 Calibrated sentiment, hate speech, stance, and custom task classifier powered by Qwen3.5-9B",
    version="2.1.0",
    lifespan=lifespan,
)

STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
async def serve_gui():
    """Serve the interactive web GUI."""
    if STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    return {"message": "Comment Classifier API running. Open /docs for Swagger UI."}


class CustomTaskDefinition(BaseModel):
    question: str = Field(..., description="The classification question to ask")
    options: List[str] = Field(..., min_length=2, max_length=8, description="Between 2 and 8 choice options")
    system_prefix: Optional[str] = Field(default=None, description="Optional system prompt prefix")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="The comment text to classify")
    post_context: Optional[str] = Field(
        default=None,
        description="Optional post caption, title, or video context to resolve sarcasm and stance",
    )
    tasks: Optional[List[str]] = Field(
        default=None,
        description="List of built-in tasks to execute. Defaults to: ['sentiment', 'hate', 'support']. Options: ['sentiment', 'sentiment_fine', 'hate', 'support', 'intent']",
    )
    target: Optional[str] = Field(
        default=None,
        description="Target entity/issue for stance detection (e.g. 'Cockroach Janta Party', 'Exam Probe', 'Modi')",
    )
    debias: Optional[bool] = Field(
        default=False,
        description="Enable Cyclic Option-Order Debiasing (runs 3 shifts to eliminate letter position bias)",
    )
    custom_tasks: Optional[Dict[str, CustomTaskDefinition]] = Field(
        default=None,
        description="Ad-hoc custom tasks with custom questions and options (up to 8 options each)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Override model (e.g. 'qwen3.5-judge' or 'comment-judge')",
    )


class AnalyzeResponse(BaseModel):
    request_id: str
    model: str
    target: Optional[str]
    post_context: Optional[str] = None
    debias_applied: bool
    elapsed_ms: float
    results: Dict[str, Any]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_comment(req: AnalyzeRequest):
    """Classify a comment across built-in or custom dynamic tasks."""
    active_clf = classifier
    if req.model and req.model != classifier.model:
        active_clf = CommentClassifier(model=req.model)

    task_names = req.tasks
    if task_names is None and not req.custom_tasks:
        task_names = ["sentiment", "hate", "support"]

    if task_names:
        for t in task_names:
            if t not in TASKS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown task '{t}'. Available built-in tasks: {list(TASKS.keys())}",
                )

    req_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    custom_dict = None
    if req.custom_tasks:
        custom_dict = {
            k: {
                "question": v.question,
                "options": v.options,
                "system_prefix": v.system_prefix or "You are an objective classifier. Answer with one letter only.",
            }
            for k, v in req.custom_tasks.items()
        }

    results = active_clf.analyze(
        req.text,
        task_names=task_names,
        target=req.target,
        debias=bool(req.debias),
        custom_tasks=custom_dict,
        temperatures=temperatures,
        thresholds=thresholds,
        post_context=req.post_context,
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Hate moderation flag
    if "hate" in results:
        results["hate"]["flag"] = results["hate"]["label"] == "hateful"

    log_entry = {
        "request_id": req_id,
        "model": active_clf.model,
        "debias": req.debias,
        "text_preview": req.text[:80],
        "elapsed_ms": elapsed_ms,
        "results": {k: v["label"] for k, v in results.items()},
    }
    request_history.append(log_entry)
    if len(request_history) > 1000:
        request_history.pop(0)

    logger.info("Req [%s]: %.1f ms | %s", req_id, elapsed_ms, log_entry["results"])
    return AnalyzeResponse(
        request_id=req_id,
        model=active_clf.model,
        target=req.target or TASKS.get("support", TaskConfig("support", "", [])).default_target,
        post_context=req.post_context,
        debias_applied=bool(req.debias),
        elapsed_ms=elapsed_ms,
        results=results,
    )


@app.get("/tasks")
async def list_tasks():
    """List all available built-in tasks and their configured options."""
    global TASKS
    TASKS = load_tasks()
    return {
        name: {
            "question": t.question,
            "options": t.options,
            "option_keys": t.option_keys,
            "examples": t.examples,
            "default_target": t.default_target,
        }
        for name, t in TASKS.items()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": classifier.model,
        "tasks": list(TASKS.keys()),
        "temperatures": temperatures,
        "total_requests": len(request_history),
    }


@app.post("/reload_config")
async def reload_config():
    """Reload calibration and prompt configurations from disk."""
    global temperatures, thresholds, TASKS
    temperatures, thresholds = load_calibration_config()
    TASKS = load_tasks()
    return {"status": "reloaded", "temperatures": temperatures, "tasks": list(TASKS.keys())}
