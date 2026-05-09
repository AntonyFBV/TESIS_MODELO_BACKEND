import os

# ─── JWT ───────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "_VVrwyAjja8D-hpblXm74Q2Yc6hkg5xU_vvSfRGecmU")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ─── BASE DE DATOS ─────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://neondb_owner:npg_m0CpE2SabQoi@ep-delicate-darkness-a8xf4x2f-pooler.eastus2.azure.neon.tech/neondb")

# ─── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = ["https://mypredictia.netlify.app"]

# ─── MODELO ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "xgboost_model_final.json")

# ─── GROQ ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"