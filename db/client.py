from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from config import settings

engine = create_async_engine(settings.db_dsn, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # 创建每设备表所需的 schema
        for schema in ("behavior", "skin_assessment", "environment"):
            await conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            )
        # 记录每个设备在 TDengine 上次处理到的时间戳
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS device_sync_state (
                device_sn         varchar(64) PRIMARY KEY,
                last_processed_ts bigint      NOT NULL DEFAULT 0,
                last_sync_at      bigint,
                updated_at        bigint
            )
        """))


async def close_db():
    await engine.dispose()


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
