import os
import sys

# ensure repo root is on sys.path so "from backend.X import Y" works
# whether running from repo root (local) or from inside backend/ (Railway)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import auth, tasks, checkins, profile, notifications
from backend.scheduler import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
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
