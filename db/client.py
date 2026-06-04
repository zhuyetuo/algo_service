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
        for schema in (
            settings.pg_schema_behavior,
            settings.pg_schema_assessment,
            settings.pg_schema_environment,
            settings.pg_schema_baseline,
        ):
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

        # 若旧表（device_sn 主键）存在则删除，以新结构重建
        await conn.execute(text("""
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'device_sync_state'
                      AND column_name = 'device_sn'
                ) THEN
                    DROP TABLE IF EXISTS processing_errors;
                    DROP TABLE IF EXISTS device_sync_state;
                END IF;
            END $$
        """))

        # 每个设备的同步断点 + 绑定用户信息
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS device_sync_state (
                device_id           bigint      PRIMARY KEY,
                user_id             bigint,
                user_timezone       varchar(32) NOT NULL DEFAULT 'UTC',
                last_processed_ts   bigint      NOT NULL DEFAULT 0,
                last_env_ts         bigint      NOT NULL DEFAULT 0,
                last_neck_temp_ts   bigint      NOT NULL DEFAULT 0,
                last_sync_at        bigint,
                updated_at          bigint
            )
        """))

        # 推理失败记录，用于重试和事后分析
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS processing_errors (
                id          bigserial   PRIMARY KEY,
                device_id   bigint      NOT NULL,
                day_ts      bigint      NOT NULL,
                error_msg   text        NOT NULL,
                retry_count int         NOT NULL DEFAULT 0,
                status      varchar(16) NOT NULL DEFAULT 'pending',
                created_at  bigint      NOT NULL,
                updated_at  bigint      NOT NULL,
                UNIQUE (device_id, day_ts)
            )
        """))

        # 若旧表（device_sn 主键）存在则删除，以新结构重建
        await conn.execute(text("""
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'pet_dog_scratch_baseline'
                      AND table_name   = 'pet_skin_baseline'
                      AND column_name  = 'device_sn'
                ) THEN
                    DROP TABLE IF EXISTS pet_dog_scratch_baseline.pet_skin_baseline;
                END IF;
            END $$
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pet_dog_scratch_baseline.pet_skin_baseline (
                device_id        bigint        PRIMARY KEY,
                baseline_mean    decimal(6,2)  NOT NULL DEFAULT 0,
                baseline_std     decimal(6,2)  NOT NULL DEFAULT 0,
                temp_coef        decimal(5,3)  NOT NULL DEFAULT 0,
                valid_days       int           NOT NULL DEFAULT 0,
                eval_phase       smallint      NOT NULL DEFAULT 0,
                confidence       decimal(4,2)  NOT NULL DEFAULT 0,
                wpeb_mean        decimal(10,4),
                wpeb_std         decimal(10,4),
                last_updated_ts  bigint,
                created_at       bigint        NOT NULL
            )
        """))


async def close_db():
    await engine.dispose()


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
