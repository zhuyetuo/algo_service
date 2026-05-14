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

    # IMU sampling rate (Hz)
    imu_sample_rate: int = 50

    # Classification sliding window
    window_seconds: int = 3
    window_overlap: float = 0.5

    # Scheduler: fetch & infer interval (minutes) — change via env var, no redeploy needed
    fetch_interval_min: int = 15

    # Scheduler cron expressions
    baseline_update_cron: str = "0 2 * * *"    # 02:00 daily
    assessment_cron: str = "0 3 * * *"          # 03:00 daily

    # Assessment — dynamic threshold phases
    # Phase 0: warm-up (days 1-3)   → no assessment
    # Phase 1: early  (days 4-14)   → z>4.0, consec>=5, avg_z>5.0
    # Phase 2: mid    (days 15-30)  → z>3.5, consec>=4, avg_z>4.0
    # Phase 3: stable (days 31+)    → z>2.5, consec>=3, avg_z>3.5
    phase1_threshold_z: float = 4.0
    phase1_threshold_consec: int = 5
    phase1_threshold_avgz: float = 5.0

    phase2_threshold_z: float = 3.5
    phase2_threshold_consec: int = 4
    phase2_threshold_avgz: float = 4.0

    phase3_threshold_z: float = 2.5
    phase3_threshold_consec: int = 3
    phase3_threshold_avgz: float = 3.5

    # Minimum std to avoid division by zero
    baseline_std_floor: float = 2.0

    # Abnormal day baseline update weight (normal days use NORMAL_W)
    abnormal_day_weight: float = 0.01
    normal_day_weight: float = 0.05

    # Night scratch window (hour range in local time, derived from UTC offset provided externally)
    night_hour_start: int = 22   # 22:00
    night_hour_end: int = 6      # 06:00

    # Minimum wear minutes to consider a day valid
    min_wear_minutes: int = 480  # 8 hours


settings = Settings()
