"""
TDengine 连接与数据拉取模块（同步，由 asyncio.to_thread 在线程池中执行）。
使用 taosrest HTTP 连接器，无需在容器内安装 TDengine 原生客户端库。
"""
import datetime
import taosrest
from config import settings


def _get_conn() -> taosrest.TaosRestConnection:
    return taosrest.connect(
        url=f"http://{settings.td_host}:{settings.td_port}",
        user=settings.td_user,
        password=settings.td_password,
    )


def _table() -> str:
    return f"{settings.td_database}.{settings.td_supertable}"


def _ts_to_ms(ts) -> int:
    if isinstance(ts, int):
        return ts
    if isinstance(ts, datetime.datetime):
        import calendar
        utc_s = calendar.timegm(ts.utctimetuple())
        return utc_s * 1000 + ts.microsecond // 1000
    return int(ts)


def td_get_devices() -> list[int]:
    """查询超级表中所有设备的 device_id 列表（调试用）。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT device_id FROM {_table()}")
        rows = cursor.fetchall()
        return [int(r[0]) for r in rows]
    finally:
        conn.close()


def td_fetch_env(device_id: int, last_ts_ms: int) -> list[dict]:
    """从 env_raw 拉取指定设备在 last_ts_ms 之后的环境数据。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        table = f"{settings.td_database}.{settings.td_supertable_env}"
        cursor.execute(f"""
            SELECT ts, env_temp, env_humi
            FROM {table}
            WHERE device_id = {device_id}
              AND ts > {last_ts_ms}
            ORDER BY ts
            LIMIT {settings.td_batch_size}
        """)
        rows = cursor.fetchall()
        return [
            {
                "ts_ms":    _ts_to_ms(r[0]),
                "env_temp": float(r[1]) if r[1] is not None else None,
                "env_humi": float(r[2]) if r[2] is not None else None,
            }
            for r in rows
        ]
    finally:
        conn.close()


def td_fetch_neck_temp(device_id: int, last_ts_ms: int) -> list[dict]:
    """从 neck_temp_raw 拉取指定设备在 last_ts_ms 之后的颈部体温数据。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        table = f"{settings.td_database}.{settings.td_supertable_neck_temp}"
        cursor.execute(f"""
            SELECT ts, neck_temp
            FROM {table}
            WHERE device_id = {device_id}
              AND ts > {last_ts_ms}
            ORDER BY ts
            LIMIT {settings.td_batch_size}
        """)
        rows = cursor.fetchall()
        return [
            {
                "ts_ms":     _ts_to_ms(r[0]),
                "neck_temp": float(r[1]) if r[1] is not None else None,
            }
            for r in rows
        ]
    finally:
        conn.close()


def td_fetch(device_id: int, last_ts_ms: int) -> list[dict]:
    """从 TDengine 拉取指定设备在 last_ts_ms 之后的新 IMU 数据。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT ts, ax, ay, az, gx, gy, gz
            FROM {_table()}
            WHERE device_id = {device_id}
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
