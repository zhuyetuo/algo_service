"""
时区名解析：把业务库里五花八门的时区写法归一到 IANA 时区。

为什么需要这个
--------------
`zoneinfo.ZoneInfo` 只认 IANA 名（Asia/Shanghai），不认缩写。业务库
`hiccpet_petos.user.timezone` 里如果存的是 "CST"、"GMT+8"、"+08:00" 这类写法，
`ZoneInfo(...)` 会直接抛 ZoneInfoNotFoundError。原来 jobs.py 里是
`except Exception: 退回 UTC`，静默吞掉——结果就是：

  - local_start / local_end 比真实本地时间**差 8 小时**
  - `_day_start_utc_ms` 按 UTC 零点分日，而不是北京零点，**日聚合边界整体错 8 小时**，
    连带影响每日汇总、抓挠计数和皮肤评估

比"报错"更糟，因为一切看起来都正常。

关于 CST 的歧义
---------------
"CST" 在全球范围是有歧义的：中国标准时间(UTC+8)、美国中部时间(UTC-6/-5)、
古巴时间(UTC-5)。本项目是国内宠物产品，默认按**中国标准时间**解释，
可通过 `CST_TIMEZONE` 环境变量改（比如真要部署到美国就改成 America/Chicago）。
"""

import logging
import re
from datetime import timedelta, timezone as dt_tz
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_logger = logging.getLogger(__name__)

# 已经警告过的名字，避免每条数据刷一行日志
_warned: set[str] = set()

# 非 IANA 的常见写法 → IANA 名
_ALIASES: dict[str, str] = {
    "PRC":            "Asia/Shanghai",
    "ASIA/BEIJING":   "Asia/Shanghai",   # 不是合法 IANA 名，但很多人这么写
    "ASIA/CHONGQING": "Asia/Shanghai",
    "BEIJING":        "Asia/Shanghai",
    "CHINA":          "Asia/Shanghai",
    "GMT":            "UTC",
    "Z":              "UTC",

    # 美国时区缩写：与 CST 不同，这些指代单一 IANA 时区，没有歧义
    # （生产库 user.timezone 里已有 America/New_York / America/Los_Angeles
    #   的用户，说明这个服务真实服务过美国用户，这几个缩写会实际出现）
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "MST": "America/Denver",
    "MDT": "America/Denver",
}

# "+08:00" / "UTC+8" / "GMT+08" / "CST-8"（POSIX 风格符号相反）等固定偏移写法
_OFFSET_RE = re.compile(
    r"^(?:UTC|GMT|CST|ETC/GMT)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?$", re.I
)


def _cst_default() -> str:
    """CST 的含义可配置，默认中国标准时间。"""
    try:
        from config import settings
        return getattr(settings, "cst_timezone", None) or "Asia/Shanghai"
    except Exception:
        return "Asia/Shanghai"


def resolve(tz_name: str | None, default: str = "UTC"):
    """
    时区名 → tzinfo 对象。解析不了时退回 default（再不行退 UTC），并打一次警告。

    支持：IANA 名、常见缩写别名、固定偏移写法（+08:00 / UTC+8 / GMT+08）。
    """
    raw = (tz_name or "").strip()
    if not raw:
        return _fallback(default)

    key = raw.upper()

    # CST 单独处理：全球有歧义，含义由配置决定
    if key == "CST":
        return _try_iana(_cst_default()) or _fallback(default)

    if key in _ALIASES:
        z = _try_iana(_ALIASES[key])
        if z:
            return z

    z = _try_iana(raw)
    if z:
        return z

    # Etc/GMT 的符号与直觉相反（Etc/GMT-8 才是 UTC+8），交给 zoneinfo 自己判断，
    # 上面 _try_iana already handled it；这里只处理纯偏移写法
    m = _OFFSET_RE.match(raw)
    if m and not key.startswith("ETC/"):
        sign = 1 if m.group(1) == "+" else -1
        hours, minutes = int(m.group(2)), int(m.group(3) or 0)
        if hours <= 14:
            return dt_tz(sign * timedelta(hours=hours, minutes=minutes))

    if raw not in _warned:
        _warned.add(raw)
        _logger.warning(
            "无法识别的时区 %r，已退回 %s。本地时间和日聚合边界会因此偏移，"
            "请修正业务库 user.timezone 或在 timezones.py 的 _ALIASES 里补一条映射。",
            raw, default,
        )
    return _fallback(default)


def canonical_name(tz_name: str | None, default: str = "UTC") -> str:
    """
    返回归一后的时区名，用于写进 user_timezone 列。

    把 "CST" 这种有歧义的写法落库成 "Asia/Shanghai"，下游读的时候不用再猜。
    """
    tz = resolve(tz_name, default)
    return getattr(tz, "key", None) or str(tz)


def _try_iana(name: str):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None


def _fallback(default: str):
    return _try_iana(default) or dt_tz.utc
