from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from api.core.config import ALLOWED_ORIGINS
from api.db.database import Base, engine
from api.routes import auth, predict

# ─── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="XGBoost Prediction API")

# ─── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ROUTERS ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(predict.router)


# ─── STARTUP ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        print("🔄 Configurando search_path...")
        await conn.execute(text("SET search_path TO public"))
        print("🔄 Verificando tablas...")
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Base de datos lista")
