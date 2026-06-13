import ssl
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from api.core.config import DATABASE_URL

# ─── SSL ───────────────────────────────────────────────────────────────────────
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# ─── ENGINE ────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    DATABASE_URL,
   echo=False,
   connect_args={"ssl": ssl_context},
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


# ─── DEPENDENCY ────────────────────────────────────────────────────────────────
async def get_db():
    async with SessionLocal() as session:
        yield session
