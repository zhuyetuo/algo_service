"""
TDengine 连接与数据拉取模块（同步，由 asyncio.to_thread 在线程池中执行）。
使用 taosrest HTTP 连接器，无需在容器内安装 TDengine 原生客户端库。
"""
import datetime
import urllib.error
import taosrest
import taosrest.errors
from config import settings


def _get_conn() -> taosrest.TaosRestConnection:
    try:
        return taosrest.connect(
            url=f"http://{settings.td_host}:{settings.td_port}",
            user=settings.td_user,
            password=settings.td_password,
        )
    except (urllib.error.URLError, OSError) as e:
        raise ConnectionError(
            f"TDengine 无法连接 {settings.td_host}:{settings.td_port}"
        ) from e


def _exec(cursor, sql: str) -> None:
    """执行 SQL，将 TDengine 业务错误统一转为 ConnectionError 以便上层简洁处理。"""
    try:
        cursor.execute(sql)
    except taosrest.errors.ConnectError as e:
        raise ConnectionError(f"TDengine 查询失败: {e}") from e


def _table() -> str:
    return f"{settings.td_database}.{settings.td_supertable}"


def _normalize_sn(device_sn: str) -> str:
    """将冒号格式（EA:CB:3E:CF:00:1D）转为 TDengine 存储格式（EA_CB_3E_CF_00_1D）。"""
    return device_sn.replace(":", "_")


def _ts_to_ms(ts) -> int:
    if isinstance(ts, int):
        return ts
    if isinstance(ts, datetime.datetime):
        import calendar
        utc_s = calendar.timegm(ts.utctimetuple())
        return utc_s * 1000 + ts.microsecond // 1000
    return int(ts)


def td_get_devices() -> list[str]:
    """查询超级表中所有设备的 device_sn 列表（调试用）。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        _exec(cursor, f"SELECT DISTINCT device_sn FROM {_table()}")
        rows = cursor.fetchall()
        return [str(r[0]).strip() for r in rows]
    finally:
        conn.close()


def td_fetch_env(device_sn: str, last_ts_ms: int) -> list[dict]:
    """从 env_data 拉取指定设备在 last_ts_ms 之后的环境数据（温湿度 + 体温）。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        table = f"{settings.td_database}.{settings.td_supertable_env}"
        _exec(cursor, f"""
            SELECT ts, temperature, humidity, body_temp
            FROM {table}
            WHERE device_sn = '{_normalize_sn(device_sn)}'
              AND ts > {last_ts_ms}
            ORDER BY ts
            LIMIT {settings.td_batch_size}
        """)
        rows = cursor.fetchall()
        return [
            {
                "ts_ms":     _ts_to_ms(r[0]),
                "env_temp":  float(r[1]) if r[1] is not None else None,
                "env_humi":  float(r[2]) if r[2] is not None else None,
                "neck_temp": float(r[3]) if r[3] is not None else None,
            }
            for r in rows
        ]
    finally:
        conn.close()


def td_fetch_neck_temp(device_sn: str, last_ts_ms: int) -> list[dict]:
    """已废弃：体温数据已合并入 env_data，请使用 td_fetch_env。"""
    return td_fetch_env(device_sn, last_ts_ms)


def td_is_charging(device_sn: str) -> bool:
    """查询设备当前是否处于充电状态（取最新一条 charging 记录）。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        table = f"{settings.td_database}.{settings.td_supertable_battery}"
        _exec(cursor, f"""
            SELECT LAST(charging)
            FROM {table}
            WHERE device_sn = '{_normalize_sn(device_sn)}'
        """)
        row = cursor.fetchone()
        return bool(row and row[0] == 1)
    finally:
        conn.close()


def td_fetch(device_sn: str, last_ts_ms: int) -> list[dict]:
    """从 TDengine 拉取指定设备在 last_ts_ms 之后的新 IMU 数据。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        _exec(cursor, f"""
            SELECT ts, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
            FROM {_table()}
            WHERE device_sn = '{_normalize_sn(device_sn)}'
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


def td_fetch_range(device_sn: str, start_ms: int, end_ms: int) -> list[dict]:
    """
    拉取指定设备在 [start_ms, end_ms) 区间内的全部 IMU 数据（离线回补专用）。

    与 td_fetch 的区别：td_fetch 是增量拉取（ts > 断点，单批上限 td_batch_size），
    这里按时间区间拉取并自动翻页，直到取完整个区间，不受单批上限限制。
    """
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        out: list[dict] = []
        cursor_ts = start_ms - 1  # 用 > 比较，减 1 以包含 start_ms 本身
        while True:
            _exec(cursor, f"""
                SELECT ts, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
                FROM {_table()}
                WHERE device_sn = '{_normalize_sn(device_sn)}'
                  AND ts > {cursor_ts} AND ts < {end_ms}
                ORDER BY ts
                LIMIT {settings.td_batch_size}
            """)
            rows = cursor.fetchall()
            if not rows:
                break
            out.extend({
                "ts_ms": _ts_to_ms(r[0]),
                "ax": float(r[1]), "ay": float(r[2]), "az": float(r[3]),
                "gx": float(r[4]), "gy": float(r[5]), "gz": float(r[6]),
            } for r in rows)
            if len(rows) < settings.td_batch_size:
                break
            cursor_ts = out[-1]["ts_ms"]
        return out
    finally:
        conn.close()


def td_device_span(device_sn: str) -> dict | None:
    """返回设备在超级表中的数据跨度：{first_ts, last_ts, count}，无数据返回 None。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        _exec(cursor, f"""
            SELECT FIRST(ts), LAST(ts), COUNT(*)
            FROM {_table()}
            WHERE device_sn = '{_normalize_sn(device_sn)}'
        """)
        rows = cursor.fetchall()
        if not rows or rows[0][0] is None:
            return None
        return {
            "first_ts": _ts_to_ms(rows[0][0]),
            "last_ts":  _ts_to_ms(rows[0][1]),
            "count":    int(rows[0][2]),
        }
    finally:
        conn.close()
