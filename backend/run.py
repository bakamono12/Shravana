"""Dev launcher — reads BACKEND_HOST/PORT from backend/.env via Settings.

Usage:
    python -m run                # from backend/
    python backend/run.py        # from repo root
    python backend/run.py --no-reload
"""
from __future__ import annotations
import argparse
import sys

import uvicorn

from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reload", action="store_true")
    parser.add_argument("--host", default=settings.BACKEND_HOST)
    parser.add_argument("--port", type=int, default=settings.BACKEND_PORT)
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


if __name__ == "__main__":
    sys.exit(main())
