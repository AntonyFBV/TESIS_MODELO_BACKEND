import asyncio
from .db import engine, Base
from .models import User

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if _name_ == "_main_":
    asyncio.run(init_models())