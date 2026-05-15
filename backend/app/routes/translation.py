from fastapi import APIRouter
from app.pipeline.translation_languages import SUPPORTED_LANGUAGES

router = APIRouter(tags=["translation"])


@router.get("/translation/languages")
async def list_languages() -> dict:
    """Return all supported translation target languages for the frontend dropdown."""
    return {
        code: {"name": info.name}
        for code, info in SUPPORTED_LANGUAGES.items()
    }
