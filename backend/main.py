import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import auth, tasks, checkins, profile, notifications
from backend.scheduler import start_scheduler, stop_scheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-load CF model + Ridge probe + feature stats + user-state
    # table so the first request doesn't pay model-load latency.
    # Failure here is non-fatal: artifacts may legitimately be absent
    # in CI / lightweight environments. The first ML call will then
    # raise FileNotFoundError with a pointer to run pretraining.
    try:
        from ml.inference.inference_api import _ensure_loaded
        _ensure_loaded()
        logger.info("ML artifacts warm-loaded.")
    except FileNotFoundError as exc:
        logger.warning("ML artifacts unavailable at startup: %s", exc)

    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(
    title="FollowThrough API",
    description="Behavioral commitment device with ML-driven personalization",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://followthrough.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(checkins.router)
app.include_router(profile.router)
app.include_router(notifications.router)


@app.get("/health")
def health():
    return {"status": "ok"}
