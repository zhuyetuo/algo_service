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

    # Logging
    log_level: str = "info"

    # LightGBM model
    model_path: str = "weights/behavior_lgbm.pkl"

    # IMU sampling rate (Hz) — used to convert seconds → samples
    imu_sample_rate: int = 50

    # Classification sliding window (seconds); keep small regardless of fetch interval
    window_seconds: int = 3
    # Overlap ratio between consecutive windows (0.0 ~ 1.0)
    window_overlap: float = 0.5

    # Scheduler: how often to fetch & infer (minutes); change without redeploying
    fetch_interval_min: int = 15

    # Scheduler: baseline & assessment cron
    baseline_update_cron: str = "0 2 * * *"   # 02:00 daily
    assessment_cron: str = "0 3 * * *"         # 03:00 daily


settings = Settings()
