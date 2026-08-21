from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=(),
    )

    # 数据库连接配置
    db_host: str = "192.168.33.253"
    db_port: int = 30100
    db_name: str = "algo"
    db_user: str = "root"
    db_password: str = "Hicc-pet-mysql-2026"

    @property
    def db_dsn(self) -> str:
        return (
            f"mysql+aiomysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?charset=utf8mb4"
        )

    # 日志级别
    log_level: str = "info"

    # 模型文件路径（imu_train 训练产出，joblib 格式）
    model_path: str = "weights/ml_rf.pkl"

    # IMU 采样率（Hz）— 设备实际上报采样率，与 imu_train 训练时 --hz 参数一致
    # 历史值: 50→20; 当前设备固件上报 25Hz，模型也以 25Hz 训练
    imu_sample_rate: int = 25

    # 分类滑动窗口配置（与 imu_train configs/data.yaml 保持一致）
    # window_seconds=2.0, stride=1.0s → overlap=0.5
    window_seconds: float = 2.0
    window_overlap: float = 0.5

    # 置信度阈值：低于此值的窗口标记为 UNKNOWN（0.0 = 禁用，直接输出模型预测）
    confidence_threshold: float = 0.0

    # 逐窗口详细推理日志（true = 每个 2s 窗口输出一行 [PC | 片上] ML=xxx）
    verbose_inference: bool = False

    # 调度器：拉取并推理的时间间隔（分钟），可通过环境变量修改，无需重新部署
    fetch_interval_sec: int = 15

    # 调度器 cron 表达式
    baseline_update_cron: str = "0 2 * * *"    # 每天 02:00 执行
    assessment_cron: str = "0 3 * * *"          # 每天 03:00 执行

    # 评估动态阈值分阶段配置
    # 阶段 0：预热期（第1-3天）  → 不做评估
    # 阶段 1：早期  （第4-14天） → z>4.0，连续>=5天，均值z>5.0
    # 阶段 2：中期  （第15-30天）→ z>3.5，连续>=4天，均值z>4.0
    # 阶段 3：稳定期（第31天起） → z>2.5，连续>=3天，均值z>3.5
    phase1_threshold_z: float = 4.0
    phase1_threshold_consec: int = 5
    phase1_threshold_avgz: float = 5.0

    phase2_threshold_z: float = 3.5
    phase2_threshold_consec: int = 4
    phase2_threshold_avgz: float = 4.0

    phase3_threshold_z: float = 2.5
    phase3_threshold_consec: int = 3
    phase3_threshold_avgz: float = 3.5

    # 基线标准差下限，防止除以零
    baseline_std_floor: float = 2.0

    # W-PEB 基线标准差下限
    baseline_std_floor_wpeb: float = 1.0

    # 夜间抓挠时间窗口（本地时间小时范围，由外部提供 UTC 偏移量推导）
    night_hour_start: int = 22   # 22:00
    night_hour_end: int = 6      # 06:00

    # 最小佩戴分钟数，低于此值视为当天无效
    min_wear_minutes: int = 480  # 8小时

    # PostgreSQL schema 名称
    pg_schema_behavior: str = "pet_dog_behavior"
    pg_schema_assessment: str = "pet_dog_skin_assessment"
    pg_schema_environment: str = "pet_dog_environment"
    pg_schema_baseline: str = "pet_dog_scratch_baseline"
    pg_schema_daily_summary: str = "pet_dog_daily_summary"

    # 业务数据库（设备绑定、用户信息来源）
    biz_schema: str = "hiccpet_petos"

    # TDengine 连接配置（REST HTTP 连接器，端口 6041）
    td_host: str = "192.168.33.253"
    td_port: int = 6041
    td_user: str = "root"
    td_password: str = "taosdata"
    td_database: str = "hiccpet_device"
    td_supertable: str = "imu_data"
    td_supertable_env: str = "env_data"
    td_supertable_neck_temp: str = "body_temp_data"
    td_supertable_battery: str = "battery_data"
    # 每次从 TDengine 单设备拉取的最大行数
    td_batch_size: int = 50000

    # 推理周期最大并发设备数（Semaphore 上限）
    device_concurrency: int = 100

    # PostgreSQL 连接池配置
    db_pool_size: int = 50       # 常驻连接数
    db_max_overflow: int = 100   # 超出 pool_size 后最多再借出的连接数


settings = Settings()
