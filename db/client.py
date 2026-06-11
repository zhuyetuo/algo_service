from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from config import settings

engine = create_async_engine(
    settings.db_dsn,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # 各业务数据库（MySQL 中 schema = database）
        for db_name in (
            settings.pg_schema_behavior,
            settings.pg_schema_assessment,
            settings.pg_schema_environment,
            settings.pg_schema_baseline,
        ):
            await conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))

        # 每个设备的同步断点 + 绑定用户信息
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS device_sync_state (
                device_id           BIGINT       PRIMARY KEY,
                user_id             BIGINT,
                user_timezone       VARCHAR(32)  NOT NULL DEFAULT 'UTC',
                last_processed_ts   BIGINT       NOT NULL DEFAULT 0,
                last_env_ts         BIGINT       NOT NULL DEFAULT 0,
                last_neck_temp_ts   BIGINT       NOT NULL DEFAULT 0,
                last_sync_at        BIGINT,
                updated_at          BIGINT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        # 推理失败记录，用于重试和事后分析
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS processing_errors (
                id          BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
                device_id   BIGINT       NOT NULL,
                day_ts      BIGINT       NOT NULL,
                error_msg   TEXT         NOT NULL,
                retry_count INT          NOT NULL DEFAULT 0,
                status      VARCHAR(16)  NOT NULL DEFAULT 'pending',
                created_at  BIGINT       NOT NULL,
                updated_at  BIGINT       NOT NULL,
                UNIQUE KEY uq_device_day (device_id, day_ts)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        await conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS `{settings.pg_schema_baseline}`.pet_skin_baseline (
                device_id        BIGINT        PRIMARY KEY,
                baseline_mean    DECIMAL(6,2)  NOT NULL DEFAULT 0,
                baseline_std     DECIMAL(6,2)  NOT NULL DEFAULT 0,
                temp_coef        DECIMAL(5,3)  NOT NULL DEFAULT 0,
                valid_days       INT           NOT NULL DEFAULT 0,
                eval_phase       SMALLINT      NOT NULL DEFAULT 0,
                confidence       DECIMAL(4,2)  NOT NULL DEFAULT 0,
                wpeb_mean        DECIMAL(10,4),
                wpeb_std         DECIMAL(10,4),
                last_updated_ts  BIGINT,
                created_at       BIGINT        NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))


async def close_db():
    await engine.dispose()


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
