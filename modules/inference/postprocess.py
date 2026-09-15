# ============================================================================
# 这个文件是 imu_train/label_service/postprocess.py 的**逐字副本**。
#
# 为什么是副本而不是 import：algo_service 是独立部署的镜像，容器里没有
# imu_train 这棵树。所以只能带一份进来。
#
# **不要在这里改。** 要改规则去改 imu_train 那边，然后重新拷贝过来。
# 两边分家之后，线上项圈报的和标注平台上看到的就不是一回事了，
# 而两边用的是同一个模型——差异全在后处理，且不会显示在任何地方。
#
# tests/unit/test_postprocess_vendored.py 会在 imu_train 存在时逐字节比对，
# 分家了当场报出来。CI 里没有 imu_train，那条会跳过——所以本机改完记得跑一次。
# ============================================================================

"""
"稳定版"后处理：把逐窗口的原始预测（调试版，模型逐窗口 argmax 的真实输出）
整理成人看着可信的片段。原始输出里活动/睡觉这类"状态"经常一秒一换来回闪，
抓挠/甩身体这类"事件"又常常是孤立的单窗口噪声，或者一次完整抓挠中间被一两个
跳成活动/甩身体的窗口切碎——这里分两套规则处理。

两种算法（算法名 = /infer 的 mode）：

stable —— 规则版
  状态类（EVENT_LABELS 之外的全部）
    1. 状态类概率向量做滑动平均（SMOOTH_WINDOWS 个窗口）再取 argmax；
    2. 连续同状态短于 MIN_STATE_S 的段并入相邻更长的那段，反复直到没有碎片。
  事件类（抓挠 / 甩身体）
    1. 双阈值滞回：该事件概率 ≥ EVENT_ENTER 进入，进入后只要 ≥ EVENT_STAY 就继续
       算同一段（中间 argmax 跳成活动的窗口抓挠概率通常还有 0.3，不会被切断）；
    2. 同类事件间隔 ≤ EVENT_GAP_S 的合成一个 bout；
    3. 抓挠 bout 前后 SHAKE_ABSORB_S 内的甩身体窗口并进抓挠（抓完常甩一下，
       模型也爱把抓挠的剧烈段判成甩身体）；
    4. bout 要么窗口数 ≥ EVENT_MIN_WINDOWS 且平均概率 ≥ EVENT_MIN_MEAN，要么最高
       概率 ≥ EVENT_SINGLE_CONF，否则丢掉；
    5. 可选：窗口自带的陀螺仪 4–8 Hz 能量占比（spec，见 pool.py）低于 SPECTRAL_MIN
       的抓挠 bout 丢掉——抓挠是后腿高频往复，频谱上有明显峰，模型之外的独立证据。

viterbi —— 稳定版 v2
  把每个窗口的各类概率当发射概率，切换类别付一个固定代价（VITERBI_SWITCH，
  对数单位），动态规划求整条时间轴代价最小的标签序列。对所有类别统一生效，
  参数只有一个；事件类出来之后同样走上面 2–5 的合并/过滤。

输入就是 /infer 返回的 windows（ts/label/conf/probs/spec），不碰模型、不碰特征，
调试版原样保留，几个版本用同一次推理的结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


@dataclass
class StableParams:
    event_labels: tuple[str, ...] = ("抓挠", "甩身体")
    scratch_label: str = "抓挠"
    shake_label: str = "甩身体"
    smooth_windows: int = 7        # 状态概率滑动平均的窗口数（stride 1s 时 ≈ 7 秒）
    min_state_s: float = 10.0      # 状态片段最短时长，短于这个并入邻居
    event_enter: float = 0.5       # 滞回：进入事件的概率门槛
    event_stay: float = 0.25       # 滞回：进入后维持在同一段的概率门槛
    event_gap_s: float = 4.0       # 同类事件之间隔多久以内合成一个 bout
    shake_absorb_s: float = 3.0    # 抓挠 bout 前后多少秒内的甩身体并进抓挠
    event_min_windows: int = 2     # bout 至少几个窗口
    event_min_mean: float = 0.45   # bout 内该事件的平均概率下限
    event_single_conf: float = 0.85  # 单窗口也保留的最高概率下限
    spectral_min: float = 0.0      # 抓挠 bout 的平均频谱占比下限，0 = 不启用
    viterbi_switch: float = 3.0    # viterbi 切换类别的代价（对数单位）
    # 疑似抓挠候选（给人工审核找漏检用，不进正式片段）
    #
    # 门槛定得太松会失去意义：0.2 进入时，一小时能抽出三百多条，绝大多数是
    # 20%~30% 的噪声，人根本审不过来，真正值得看的那几条反而被淹掉。这里的
    # 目标不是"把所有可能都列出来"，而是"给人一份一小时能看完的清单"。
    cand_enter: float = 0.3        # 低门槛滞回：进入（比正式的 0.5 低，但不能太低）
    # 维持门槛不能比背景噪声低太多：背景在 0.05~0.28 晃时，stay=0.2 会让一段真抓挠
    # 顺着噪声一路延伸出去，边界拖长、整段平均被稀释（实测 0.57 掉到 0.42），
    # 反而在按置信度排序时沉下去。0.25 刚好卡在噪声上沿之上
    cand_stay: float = 0.25        # 低门槛滞回：维持
    cand_min_windows: int = 2
    cand_min_mean: float = 0.3     # 整段的平均概率也要够，挡掉"就一个窗口冒了一下"
    # 只靠频谱把一段拎成候选：默认 0 = 关掉（见 scratch_candidates 里的说明，
    # 实测这批设备的本底就偏高，抽出来几乎全是模型置信 0% 的噪声）
    cand_spec_min: float = 0.0
    # 每个文件最多给这么多条（在 app.py 里按置信度从高到低裁，并记日志说明丢了多少）
    cand_max: int = 40
    # 边界微调：在片段起止各 ±refine_margin_s 内按陀螺仪能量找真正的起止
    refine_margin_s: float = 1.0
    refine_ratio: float = 0.3      # 能量高于片段内中位数 × 这个比例才算"在动"


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    for f in (_TS_FMT, "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    return None


def _fmt_ts(t: datetime) -> str:
    return t.strftime(_TS_FMT)[:-3]


def _zones(ts: list[datetime], window_s: float, stride_s: float, label_mode: str) -> list[tuple[datetime, datetime]]:
    """每个窗口在时间轴上负责的区间，跟 infer_csv_scratch.window_zone 一致。"""
    out = []
    n = len(ts)
    for k in range(n):
        if label_mode == "center":
            c = ts[k] + timedelta(seconds=window_s / 2)
            out.append((c - timedelta(seconds=stride_s / 2), c + timedelta(seconds=stride_s / 2)))
        else:
            end = ts[k + 1] if k + 1 < n else ts[k] + timedelta(seconds=window_s)
            out.append((ts[k], end))
    return out


def _runs(labels: list[str]) -> list[list]:
    """[[label, i0, i1], ...] 连续同标签的区间（闭区间，窗口下标）"""
    runs: list[list] = []
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i
        else:
            runs.append([lab, i, i])
    return runs


def _absorb_short_runs(labels: list[str], zones, min_s: float) -> list[str]:
    """把短于 min_s 的状态片段并进相邻更长的那段，直到没有碎片。"""
    labels = list(labels)
    while True:
        runs = _runs(labels)
        if len(runs) <= 1:
            return labels
        dur = [(zones[i1][1] - zones[i0][0]).total_seconds() for _, i0, i1 in runs]
        short = [k for k, d in enumerate(dur) if d < min_s]
        if not short:
            return labels
        k = min(short, key=lambda kk: dur[kk])
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k + 1 < len(runs) else None
        target = right if left is None else left if right is None else (left if dur[k - 1] >= dur[k + 1] else right)
        for i in range(runs[k][1], runs[k][2] + 1):
            labels[i] = target[0]


def _smooth_states(probs: list[dict], states: list[str], n: int, k: int) -> list[str]:
    half = max(0, k // 2)
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        best, best_v = None, -1.0
        for lab in states:
            v = sum(probs[j].get(lab, 0.0) for j in range(lo, hi)) / (hi - lo)
            if v > best_v:
                best, best_v = lab, v
        out.append(best or (states[0] if states else ""))
    return out


def _viterbi(probs: list[dict], classes: list[str], switch_cost: float) -> list[str]:
    """发射 = log(prob)，切换类别扣 switch_cost，求整条序列代价最小的标签。"""
    n, m = len(probs), len(classes)
    if n == 0 or m == 0:
        return []
    eps = 1e-6
    emit = [[math.log(max(eps, p.get(c, 0.0))) for c in classes] for p in probs]
    score = list(emit[0])
    back: list[list[int]] = []
    for i in range(1, n):
        best_prev = max(range(m), key=lambda j: score[j])
        new_score, bp = [], []
        for c in range(m):
            stay = score[c]
            switch = score[best_prev] - switch_cost
            if stay >= switch:
                new_score.append(stay + emit[i][c]); bp.append(c)
            else:
                new_score.append(switch + emit[i][c]); bp.append(best_prev)
        score, back = new_score, back + [bp]
    cur = max(range(m), key=lambda j: score[j])
    path = [cur]
    for bp in reversed(back):
        cur = bp[cur]
        path.append(cur)
    path.reverse()
    return [classes[c] for c in path]


def _bouts_hysteresis(pv: list[float], enter: float, stay: float) -> list[list[int]]:
    """双阈值滞回：≥enter 进入，≥stay 维持。返回 [[i0..i1 的下标列表], ...]"""
    bouts: list[list[int]] = []
    cur: list[int] | None = None
    for i, v in enumerate(pv):
        if cur is None:
            if v >= enter:
                cur = [i]
        else:
            if v >= stay:
                cur.append(i)
            else:
                bouts.append(cur); cur = None
    if cur:
        bouts.append(cur)
    return bouts


def _merge_gaps(bouts: list[list[int]], zones, gap_s: float) -> list[list[int]]:
    merged: list[list[int]] = []
    for b in bouts:
        if merged and (zones[b[0]][0] - zones[merged[-1][-1]][1]).total_seconds() <= gap_s:
            merged[-1] = list(range(merged[-1][0], b[-1] + 1))
        else:
            merged.append(list(range(b[0], b[-1] + 1)))
    return merged


# ── 掉数据（MISSING）────────────────────────────────────────────────────
#
# 采集端蓝牙断了会把六个轴全写 0 来占帧对齐的位置（也可能写 MISSING → NaN）。
# 真实佩戴时六轴不可能同时精确为 0——重力总落在某个轴上——所以"六轴全 0"就是
# 这一段根本没有数据。这种窗口喂给模型只会得到一个凭空的类别（实测多半判成
# 睡觉），既不能算进有效佩戴，也绝不能进训练集。
#
# 处理办法是"挖洞"而不是"丢行"：CSV 里的行照留（不然工作台上看视频会跳帧、
# 波形和时间轴对不上），只是把这些时间段从预测结果里抠掉。
def missing_spans(windows: list[dict], window_s: float, stride_s: float, label_mode: str,
                  min_ratio: float = 0.5) -> list[tuple]:
    """窗口里缺数据比例 ≥ min_ratio 的，合并成一组不相交的 (start, end) 时间区间。"""
    ts = [_parse_ts(w.get("ts")) for w in windows]
    if not ts or any(t is None for t in ts):
        return []
    zones = _zones(ts, window_s, stride_s, label_mode)  # type: ignore[arg-type]
    out: list[list] = []
    for w, (a, b) in zip(windows, zones):
        m = w.get("missing")
        if m is None or m < min_ratio:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def drop_missing(segments: list[dict], holes: list[tuple]) -> list[dict]:
    """把片段里跟 holes 重叠的部分抠掉；一段被从中间挖开就拆成两段。"""
    if not holes:
        return segments
    out: list[dict] = []
    for seg in segments:
        a, b = _parse_ts(seg.get("start_ts")), _parse_ts(seg.get("end_ts"))
        if a is None or b is None:
            out.append(seg)
            continue
        pieces = [(a, b)]
        for h0, h1 in holes:
            nxt = []
            for x, y in pieces:
                if h1 <= x or h0 >= y:
                    nxt.append((x, y))
                    continue
                if h0 > x:
                    nxt.append((x, h0))
                if h1 < y:
                    nxt.append((h1, y))
            pieces = nxt
        for x, y in pieces:
            # 挖剩不足一个窗口的碎片没有意义，丢掉
            if (y - x).total_seconds() < 1.0:
                continue
            piece = dict(seg)
            piece["start_ts"] = _fmt_ts(x)
            piece["end_ts"] = _fmt_ts(y)
            out.append(piece)
    return out


def scratch_candidates(windows: list[dict], final_segments: dict[str, list[dict]],
                       window_s: float, stride_s: float, label_mode: str,
                       params: StableParams | None = None) -> list[dict]:
    """
    疑似抓挠候选：正式片段（稳定版/v2）为了准把 20% 左右的真抓挠也滤掉了，这里用
    低门槛再抽一遍给人工看——两类来源：
      low_conf：抓挠概率 ≥ cand_enter 进入、≥ cand_stay 维持，≥ cand_min_windows 个窗口；
      spectral：陀螺仪 4–8 Hz 占比 ≥ cand_spec_min 且连续 ≥ cand_min_windows 个窗口，
                模型没判成抓挠（模型漏检但物理信号像抓挠）。
    跟正式抓挠片段重叠的去掉。返回结构同 segments 元素，多 reason 字段。
    """
    p = params or StableParams()
    n = len(windows)
    if n == 0:
        return []
    ts = [_parse_ts(w.get("ts")) for w in windows]
    if any(t is None for t in ts):
        return []
    zones = _zones(ts, window_s, stride_s, label_mode)  # type: ignore[arg-type]
    sl = p.scratch_label
    probs = [w.get("probs") or {} for w in windows]
    spec = [w.get("spec") for w in windows]
    pv = [probs[i].get(sl, 0.0) for i in range(n)]

    taken = [False] * n
    for seg in final_segments.get(sl) or []:
        s0, e0 = _parse_ts(seg["start_ts"]), _parse_ts(seg["end_ts"])
        for i in range(n):
            if s0 and e0 and zones[i][0] < e0 and zones[i][1] > s0:
                taken[i] = True

    cands: list[tuple[list[int], str]] = []
    for b in _merge_gaps(_bouts_hysteresis(pv, p.cand_enter, p.cand_stay), zones, p.event_gap_s):
        b = [i for i in b if not taken[i]]
        if len(b) >= p.cand_min_windows:
            cands.append((b, "low_conf"))
    # 频谱这一路默认关掉（cand_spec_min <= 0）。当初的设想是"模型之外的独立证据"，
    # 但实测下来对这批设备完全不具备区分度：抽出来的几百条里绝大多数模型置信度是
    # 0%、频谱占比却有 0.5~0.68——这只狗的陀螺仪本底在 4–8 Hz 就偏高，阈值定在哪
    # 都是一片噪声。spec 值仍然照常算、照常在界面上显示，可以用来辅助判断某一条
    # 候选像不像；只是不再单独凭它把一段拎出来。哪天换了设备想试，把
    # CAND_SPEC_MIN 设成 >0 即可。
    if p.cand_spec_min > 0:
        covered = {i for b, _ in cands for i in b}
        run: list[int] = []
        for i in range(n + 1):
            ok = i < n and not taken[i] and i not in covered and spec[i] is not None and spec[i] >= p.cand_spec_min
            if ok:
                run.append(i)
            else:
                if len(run) >= p.cand_min_windows:
                    cands.append((list(run), "spectral"))
                run = []

    out = []
    for b, reason in cands:
        i0, i1 = b[0], b[-1]
        vals = [pv[i] for i in range(i0, i1 + 1)]
        mean_c = sum(vals) / len(vals)
        # 低置信那一路要求整段平均够高；频谱那一路本来就是靠物理证据进来的，不看概率
        if reason == "low_conf" and mean_c < p.cand_min_mean:
            continue
        sv = [spec[i] for i in range(i0, i1 + 1) if spec[i] is not None]
        out.append({
            "start_ts": _fmt_ts(zones[i0][0]),
            "end_ts": _fmt_ts(zones[i1][1]),
            "conf_max": float(max(vals)),
            "conf_mean": float(sum(vals) / len(vals)),
            "n_windows": i1 - i0 + 1,
            "spec": round(sum(sv) / len(sv), 3) if sv else None,
            "reason": reason,
        })
    # 置信度高的排前面：那些多半是被平滑抹掉的真抓挠，最值得先看。
    # 数量上限不在这里砍——裁剪和"丢了多少条"的日志一起放在调用方（app.py），
    # 免得这里静悄悄少给几条、外面还不知道
    out.sort(key=lambda c: c["conf_mean"], reverse=True)
    return out


def refine_boundaries(segments: list[dict], envelope: dict | None, params: StableParams | None = None) -> None:
    """
    用陀螺仪能量包络（pool.py 算的，10 Hz）把片段起止从 1 秒窗口对齐精确到 0.1 秒：
    在原起点 ±margin 内找第一个能量高于阈值的点当新起点，终点同理找最后一个。
    阈值 = 片段内能量中位数 × refine_ratio。原地改 segments 的 start_ts/end_ts。
    """
    p = params or StableParams()
    if not envelope or not segments:
        return
    t0 = _parse_ts(envelope.get("t0"))
    hz = float(envelope.get("hz") or 0)
    energy = envelope.get("energy") or []
    if t0 is None or hz <= 0 or not energy:
        return
    n = len(energy)

    def idx(t: datetime) -> int:
        return int(round((t - t0).total_seconds() * hz))

    def at(i: int) -> datetime:
        return t0 + timedelta(seconds=i / hz)

    m = int(round(p.refine_margin_s * hz))
    for seg in segments:
        s, e = _parse_ts(seg["start_ts"]), _parse_ts(seg["end_ts"])
        if s is None or e is None:
            continue
        si, ei = idx(s), idx(e)
        inner = sorted(energy[max(0, si):min(n, ei)])
        if not inner:
            continue
        thr = inner[len(inner) // 2] * p.refine_ratio
        lo, hi = max(0, si - m), min(n - 1, si + m)
        new_s = next((i for i in range(lo, hi + 1) if energy[i] >= thr), None)
        lo, hi = max(0, ei - m), min(n - 1, ei + m)
        new_e = next((i for i in range(hi, lo - 1, -1) if energy[i] >= thr), None)
        if new_s is not None and new_e is not None and new_e > new_s:
            seg["start_ts"], seg["end_ts"] = _fmt_ts(at(new_s)), _fmt_ts(at(new_e + 1))


def stabilize(windows: list[dict], classes: list[str], target_labels: list[str],
              window_s: float, stride_s: float, label_mode: str,
              params: StableParams | None = None, algo: str = "stable") -> dict[str, list[dict]]:
    """返回 {label: [{start_ts, end_ts, conf_max, conf_mean, n_windows, spec}]}，跟调试版 segments 同结构。"""
    p = params or StableParams()
    n = len(windows)
    if n == 0:
        return {lab: [] for lab in target_labels}
    ts = [_parse_ts(w.get("ts")) for w in windows]
    if any(t is None for t in ts):
        return {lab: [] for lab in target_labels}
    zones = _zones(ts, window_s, stride_s, label_mode)  # type: ignore[arg-type]

    events = [lab for lab in p.event_labels if lab in classes]
    states = [lab for lab in classes if lab not in events]
    probs = [w.get("probs") or {} for w in windows]
    spec = [w.get("spec") for w in windows]
    raw_label = [w.get("label") for w in windows]

    # ── 状态时间轴 ──
    if algo == "viterbi":
        decoded = _viterbi(probs, classes, p.viterbi_switch)
        # 状态部分：把解码出的事件窗口先按邻居填掉，得到纯状态轴
        state_label = []
        last = next((d for d in decoded if d in states), states[0] if states else "")
        for d in decoded:
            if d in states:
                last = d
            state_label.append(last)
    else:
        decoded = None
        state_label = _absorb_short_runs(_smooth_states(probs, states, n, p.smooth_windows), zones, p.min_state_s)

    # ── 事件 bout ──
    final = list(state_label)
    bouts_by_event: dict[str, list[list[int]]] = {}
    for ev in events:
        pv = [probs[i].get(ev, 0.0) for i in range(n)]
        if decoded is not None:
            idx = [i for i in range(n) if decoded[i] == ev]
            raw_b = [[i] for i in idx]
        else:
            raw_b = _bouts_hysteresis(pv, p.event_enter, p.event_stay)
        bouts_by_event[ev] = _merge_gaps(raw_b, zones, p.event_gap_s)

    # 抓挠吞并前后的甩身体窗口
    sl, sh = p.scratch_label, p.shake_label
    if sl in bouts_by_event and sh in classes:
        shake_idx = {i for i in range(n) if raw_label[i] == sh or (decoded is not None and decoded[i] == sh)}
        grown = []
        for b in bouts_by_event[sl]:
            i0, i1 = b[0], b[-1]
            while i0 - 1 >= 0 and (i0 - 1) in shake_idx and (zones[i0][0] - zones[i0 - 1][1]).total_seconds() <= p.shake_absorb_s:
                i0 -= 1
            while i1 + 1 < n and (i1 + 1) in shake_idx and (zones[i1 + 1][0] - zones[i1][1]).total_seconds() <= p.shake_absorb_s:
                i1 += 1
            grown.append(list(range(i0, i1 + 1)))
        bouts_by_event[sl] = _merge_gaps(grown, zones, p.event_gap_s)
        taken = {i for b in bouts_by_event[sl] for i in b}
        if sh in bouts_by_event:
            bouts_by_event[sh] = [[i for i in b if i not in taken] for b in bouts_by_event[sh]]
            bouts_by_event[sh] = [b for b in bouts_by_event[sh] if b]

    # 门槛过滤 + 写到时间轴（抓挠优先于甩身体）
    for ev in sorted(bouts_by_event, key=lambda e: 0 if e == sl else 1):
        for b in bouts_by_event[ev]:
            pv = [probs[i].get(ev, 0.0) for i in b]
            mean_c, max_c = sum(pv) / len(pv), max(pv)
            keep = (len(b) >= p.event_min_windows and mean_c >= p.event_min_mean) or max_c >= p.event_single_conf
            if keep and ev == sl and p.spectral_min > 0:
                sv = [spec[i] for i in b if spec[i] is not None]
                if sv and sum(sv) / len(sv) < p.spectral_min:
                    keep = False
            if not keep:
                continue
            for i in b:
                if final[i] not in events or ev == sl:
                    final[i] = ev

    # ── 按最终标签切片段 ──
    out: dict[str, list[dict]] = {lab: [] for lab in target_labels}
    for lab, i0, i1 in _runs(final):
        if lab not in out:
            continue
        pv = [probs[i].get(lab, 0.0) for i in range(i0, i1 + 1)]
        sv = [spec[i] for i in range(i0, i1 + 1) if spec[i] is not None]
        out[lab].append({
            "start_ts": _fmt_ts(zones[i0][0]),
            "end_ts": _fmt_ts(zones[i1][1]),
            "conf_max": float(max(pv)) if pv else 0.0,
            "conf_mean": float(sum(pv) / len(pv)) if pv else 0.0,
            "n_windows": i1 - i0 + 1,
            "spec": round(sum(sv) / len(sv), 3) if sv else None,
        })
    return out
