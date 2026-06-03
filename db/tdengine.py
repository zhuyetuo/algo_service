"""
TDengine 连接与数据拉取模块（同步，由 asyncio.to_thread 在线程池中执行）。
使用 taosrest HTTP 连接器，无需在容器内安装 TDengine 原生客户端库。
"""
import datetime
import taosrest
from config import settings


def _get_conn() -> taosrest.TaosRestConnection:
    """创建并返回 TDengine REST 连接（HTTP，端口 6041）。"""
    return taosrest.connect(
        url=f"http://{settings.td_host}:{settings.td_port}",
        user=settings.td_user,
        password=settings.td_password,
    )


def _table() -> str:
    """返回全限定超级表名，避免 taosrest 不传 database 参数的问题。"""
    return f"{settings.td_database}.{settings.td_supertable}"


def _ts_to_ms(ts) -> int:
    """将 TDengine 返回的 ts 字段转换为 UTC 毫秒时间戳。"""
    if isinstance(ts, int):
        return ts
    if isinstance(ts, datetime.datetime):
        import calendar
        utc_s = calendar.timegm(ts.utctimetuple())
        return utc_s * 1000 + ts.microsecond // 1000
    return int(ts)


def td_get_devices() -> list[str]:
    """
    查询超级表中所有设备的 device_sn 列表。
    用于初始化 device_sync_state。
    """
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT device_sn FROM {_table()}")
        rows = cursor.fetchall()
        return [str(r[0]).strip() for r in rows]
    finally:
        conn.close()


def td_fetch(device_sn: str, last_ts_ms: int) -> list[dict]:
    """
    从 TDengine 拉取指定设备在 last_ts_ms 之后的新 IMU 数据。
    返回列表，每个元素为包含 ts_ms 和 6 轴数据的字典。
    无新数据时返回空列表。
    """
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        # 使用超级表按 device_sn tag 过滤，效率等同于直接查子表
        cursor.execute(f"""
            SELECT ts, ax, ay, az, gx, gy, gz
            FROM {_table()}
            WHERE device_sn = '{device_sn}'
              AND ts > {last_ts_ms}
            ORDER BY ts
            LIMIT {settings.td_batch_size}
        """)
        rows = cursor.fetchall()
        return [
            {
                "ts_ms": _ts_to_ms(r[0]),
                "ax": float(r[1]),
                "ay": float(r[2]),
                "az": float(r[3]),
                "gx": float(r[4]),
                "gy": float(r[5]),
                "gz": float(r[6]),
            }
            for r in rows
        ]
    finally:
        conn.close()
