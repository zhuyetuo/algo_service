"""把逐窗口的概率过一遍「稳定版 v2」后处理，再转回行为事件。

原来这条线是：argmax → 滑动多数票(k=5) → 连续同标签合并成事件。
问题是模型逐窗口的输出是抖的，一次连续的抓挠会被拆成好几段——
统计出来的"今天抓了几次"整个偏大，而每一条记录看起来都正常。

稳定版 v2 做的是另一件事：把整段时间轴放在一起解码（viterbi，切换类别
要付代价），再按规则把事件段合并、过滤。标注平台上效果最好的就是这套，
线上项圈跟平台用的是同一个模型，差异全在后处理这一层。

这里只做**转换**，规则一条都不重写——规则在 postprocess.py 里，
那是 imu_train/label_service 那份的逐字副本。
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

from modules.inference import postprocess as P

# postprocess 用字符串时间戳，格式固定
_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


def _fmt(ms: int) -> str:
    return _dt.datetime.fromtimestamp(ms / 1000.0, _dt.timezone.utc).strftime(_TS_FMT)[:-3]


def _parse(s: str) -> int:
    d = _dt.datetime.strptime(s, _TS_FMT).replace(tzinfo=_dt.timezone.utc)
    return int(d.timestamp() * 1000)


def build_windows(proba: np.ndarray, classes: list[str], base_ts_ms: int,
                  step_samples: int, fs: int) -> list[dict]:
    """逐窗口概率 → postprocess.stabilize 吃的那种 windows 列表。

    每个窗口的 ts 是**这个窗口起点**的时间，跟 imu_train 的
    infer_csv_scratch 一致（那边 zones 就是按 ts[k]..ts[k+1] 算的）。
    用窗口中点的话所有片段的时间会整体偏半个窗口，而每一段看着都正常。
    """
    out = []
    for i in range(len(proba)):
        p = proba[i]
        ts_ms = base_ts_ms + int(i * step_samples / fs * 1000)
        d = {classes[c]: float(p[c]) for c in range(len(classes))}
        out.append({
            "ts": _fmt(ts_ms),
            "probs": d,
            # raw_label 是**没经过 viterbi 的 argmax**。postprocess 里
            # "抓挠吞并前后的甩身体"那一步会用到它（raw_label 或 decoded
            # 命中都算），所以必须给对
            "label": classes[int(np.argmax(p))],
            "spec": None,
        })
    return out


def events_from_segments(segments: dict, label_to_code: dict,
                         window_s: float) -> list[dict]:
    """stabilize 的片段 → 跟 windows_to_events 一样结构的事件。

    字段名和类型保持不变，写库那段一个字都不用改——
    改了的话表结构、下游日汇总、APP 全要跟着动。
    """
    out = []
    for lab, items in (segments or {}).items():
        code = label_to_code.get(lab)
        if code is None:
            # 模型有这个类别但业务侧没有对应编码：**跳过而不是塞个 0**。
            # 塞 0 的话库里会多出一批"未知"事件，看起来像识别失败
            continue
        for s in items:
            out.append({
                "behavior_type": int(code),
                "start_time": _parse(s["start_ts"]),
                "end_time": _parse(s["end_ts"]),
                # conf_mean 而不是 conf_max：原来那条线写的就是段内平均
                # （windows_to_events 里 cur_conf_sum / cur_conf_cnt），
                # 换成 max 的话库里的置信度会整体抬高，跟历史数据不可比
                "confidence": round(float(s.get("conf_mean") or 0.0), 4),
            })
    # 按开始时间排序：stabilize 是按类别分组返回的，不排的话写库顺序是乱的。
    # ts_start 上有唯一索引，乱序不会错，但日志和排查时很难看
    out.sort(key=lambda e: (e["start_time"], e["behavior_type"]))
    return out


def stabilize_events(proba: np.ndarray, classes: list[str], label_to_code: dict,
                     base_ts_ms: int, window_samples: int, step_samples: int,
                     fs: int, params: P.StableParams | None = None,
                     algo: str = "viterbi") -> list[dict]:
    """整条：概率 → 稳定版 v2 → 行为事件。"""
    if proba is None or len(proba) == 0:
        return []
    window_s = window_samples / float(fs)
    stride_s = step_samples / float(fs)
    windows = build_windows(proba, classes, base_ts_ms, step_samples, fs)
    segments = P.stabilize(
        windows, classes, classes, window_s, stride_s,
        # label_mode 跟 imu_train 推理时一致（majority），
        # 决定每个窗口在时间轴上负责哪一段
        "majority", params or P.StableParams(), algo=algo,
    )
    return events_from_segments(segments, label_to_code, window_s)
