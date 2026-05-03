from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import init_db
from app.ml.downloader import start_background_downloads
from app.routes import upload, jobs, subtitles, models_admin, ws, translation, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await start_background_downloads()

    from app.config import settings
    if settings.REMOTE_GPU_URL:
        from app.services.remote_health import start_health_poller
        import asyncio
        asyncio.create_task(start_health_poller())

    yield


app = FastAPI(title="Shravana", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(subtitles.router, prefix="/api")
app.include_router(models_admin.router, prefix="/api")
app.include_router(ws.router, prefix="/api")
app.include_router(translation.router, prefix="/api")
app.include_router(system.router, prefix="/api")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


# Serve built frontend in production mode
if settings.SHRAVANA_SERVE_UI:
    ui_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if ui_dist.exists():
        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")
