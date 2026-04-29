from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

from backend.routes import api as api_module
from backend.db.database import init_db
from backend.graph.neo4j_client import init_neo4j_schema
from backend.rag.rag_utils import GRAPH_ENABLED

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# CORS 配置：从环境变量读取允许的源列表
# 开发环境使用 "*"，生产环境必须指定具体域名
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


def create_app() -> FastAPI:
    app = FastAPI(title="Cute Cat Bot API")

    @app.on_event("startup")
    async def _startup_init_db():
        init_db()
        if GRAPH_ENABLED:
            init_neo4j_schema()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # No-cache middleware for development
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(api_module.router)

    # serve frontend static files at root
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
