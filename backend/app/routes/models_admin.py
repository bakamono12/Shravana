from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models_db import ModelDownload
from app.schemas import ModelDownloadOut

router = APIRouter()


@router.get("/models", response_model=list[ModelDownloadOut])
async def list_models(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ModelDownload).order_by(ModelDownload.name))
    return result.scalars().all()
