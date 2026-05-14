from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    db_host: str = "postgres"
    db_port: int = 5432
    db_name: str = "algo"
    db_user: str = "algo"
    db_password: str = "algo"

    @property
    def db_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Logging
    log_level: str = "info"

    # Inference
    model_weights_path: str = "weights/behavior_model.pt"
    inference_device: str = "cpu"  # "cuda" if GPU available

    # Scheduler
    baseline_update_cron: str = "0 2 * * *"   # 02:00 daily
    assessment_cron: str = "0 3 * * *"         # 03:00 daily


settings = Settings()
