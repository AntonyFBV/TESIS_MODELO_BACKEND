from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import ssl

#DATABASE_URL = "postgresql+asyncpg://postgres:Fabio123@localhost:5432/modelo_empresas"

DATABASE_URL = "postgresql+asyncpg://neondb_owner:npg_m0CpE2SabQoi@ep-delicate-darkness-a8xf4x2f-pooler.eastus2.azure.neon.tech/neondb"

# Configurar SSL para Neon
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"ssl": ssl_context}
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()