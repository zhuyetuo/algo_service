"""稳定版 v2 后处理：副本没分家 + 转换层对不对。

这些测试**不需要 sklearn / joblib**——它们验的是后处理和转换，
不是模型。装不上依赖的机器上也要能跑，否则最该被验的那部分反而没人验。
"""

import datetime as _dt
import os

import numpy as np
import pytest

from modules.inference import postprocess as P
from modules.inference import stabilize as S

CLASSES = ["抓挠", "活动", "睡觉"]
CODE = {"抓挠": 3, "活动": 1, "睡觉": 2}
FS, WIN, STEP = 25, 50, 25          # 2s 窗口、1s 步长，跟 config 默认一致
BASE = 1_757_000_000_000


def _legacy():
    """老路子那两个函数。model.py 顶层 import joblib/sklearn，而这两个函数
    根本不用它们——打个桩拿到真代码，比自己照抄一份可靠。
    （装不上依赖的机器上，最该被验的对比反而跑不了，那就本末倒置了。）"""
    import sys
    import types
    for name in ("joblib", "sklearn"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                sys.modules[name] = types.SimpleNamespace(
                    load=lambda *a, **k: (_ for _ in ()).throw(
                        AssertionError("这两个函数不该用到 joblib")))
    from modules.inference.model import _majority_smooth, windows_to_events
    return _majority_smooth, windows_to_events


def _probs(spec, conf=0.95):
    """按 [(类别下标, 窗口数), ...] 造干净的概率序列。"""
    rows = []
    for c, k in spec:
        for _ in range(k):
            p = np.full(len(CLASSES), (1.0 - conf) / (len(CLASSES) - 1), np.float32)
            p[c] = conf
            rows.append(p)
    return np.array(rows, np.float32)


# ── 副本没分家 ────────────────────────────────────────────────────────────


def test_vendored_postprocess_is_byte_identical_to_imu_train():
    """副本必须跟 imu_train/label_service/postprocess.py 逐字节一样。

    **这是整件事的地基**：两边分家之后，线上项圈报的和标注平台上看到的
    就不是一回事了，而两边用的是同一个模型——差异全在后处理，
    且不会显示在任何地方。

    CI 里没有 imu_train，这条会跳过。所以在有 imu_train 的机器上改完
    记得跑一次。
    """
    src = os.path.expanduser(
        os.environ.get("IMU_TRAIN", "~/imu_train") + "/label_service/postprocess.py")
    if not os.path.exists(src):
        pytest.skip(f"找不到 {src}，没法比对副本")
    here = os.path.join(os.path.dirname(__file__), "..", "..",
                        "modules", "inference", "postprocess.py")
    with open(src, encoding="utf-8") as f:
        want = f.read()
    with open(here, encoding="utf-8") as f:
        got = f.read()
    # 副本开头多了一段说明，去掉再比
    marker = "# ============================================================================\n\n"
    idx = got.rfind(marker)
    assert idx >= 0, "副本头部的说明被删了，那就没人知道这是副本、不能直接改"
    body = got[idx + len(marker):]
    assert body == want, (
        "副本跟 imu_train 那份不一样了。要改规则去改 imu_train，"
        "再重新拷过来——别在这边改。")


# ── 时间戳往返 ────────────────────────────────────────────────────────────


def test_timestamp_roundtrip_is_exact_to_the_millisecond():
    """毫秒进、毫秒出。**差一毫秒都不行**——事件的 ts_start 上有唯一索引，
    时间对不上就是重复写入或者漏写，而两者都不报错。"""
    for ms in (BASE, BASE + 1, BASE + 999, BASE + 86_399_999):
        assert S._parse(S._fmt(ms)) == ms


def test_window_ts_is_the_window_start_not_the_middle():
    """窗口的 ts 取起点，跟 imu_train 的 zones 一致。

    取中点的话所有片段的时间整体偏半个窗口（这里是 1 秒），
    而每一段看着都正常——最难发现的那种错。
    """
    proba = _probs([(1, 3)])
    w = S.build_windows(proba, CLASSES, BASE, STEP, FS)
    assert S._parse(w[0]["ts"]) == BASE
    assert S._parse(w[1]["ts"]) == BASE + 1000      # step=25 @25Hz = 1s
    assert S._parse(w[2]["ts"]) == BASE + 2000


def test_raw_label_is_the_argmax():
    """windows 里的 label 要是没经过 viterbi 的 argmax——
    postprocess 的"抓挠吞并前后的甩身体"那一步会用它。"""
    proba = _probs([(0, 1), (2, 1)])
    w = S.build_windows(proba, CLASSES, BASE, STEP, FS)
    assert [x["label"] for x in w] == ["抓挠", "睡觉"]


# ── 整条：概率 → 事件 ─────────────────────────────────────────────────────


def test_events_have_the_same_shape_as_the_legacy_path():
    """字段名和类型要跟 windows_to_events 一样，写库那段一个字都不用改。"""
    proba = _probs([(1, 20), (0, 6), (1, 20)])
    evs = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS)
    assert evs, "一个事件都没有，这条测试在空转"
    for e in evs:
        assert set(e) == {"behavior_type", "start_time", "end_time", "confidence"}
        assert isinstance(e["behavior_type"], int)
        assert isinstance(e["start_time"], int) and isinstance(e["end_time"], int)
        assert e["end_time"] > e["start_time"]
        assert 0.0 <= e["confidence"] <= 1.0


def test_events_are_sorted_by_time():
    """stabilize 按类别分组返回，不排的话写库顺序是乱的。"""
    proba = _probs([(1, 15), (0, 5), (2, 15), (0, 5), (1, 15)])
    evs = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS)
    assert evs == sorted(evs, key=lambda e: (e["start_time"], e["behavior_type"]))


def test_v2_merges_what_the_legacy_path_would_split():
    """**这就是换这套后处理的理由**：中间断 3 秒的一次抓挠要算一次，不是两次。

    老路子（滑动多数票 k=5 + 连续同标签合并）只够填掉 1~2 个窗口的断口；
    断 3 个窗口它就报成两次，统计出来的"今天抓了几次"偏大。
    v2 的 event_gap_s=4s 能把它合起来。

    间隔用 3 是量出来的，不是随手挑的——1~2 窗老路子自己就合上了，
    拿那个当例子的话两边结果一样，这条测试什么都验不到（第一版就是 2）。
    """
    _majority_smooth, windows_to_events = _legacy()

    # 抓挠 4 窗 → 活动 3 窗（3s）→ 抓挠 4 窗
    proba = _probs([(1, 20), (0, 4), (1, 3), (0, 4), (1, 20)])

    v2 = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS)
    n_v2 = len([e for e in v2 if e["behavior_type"] == CODE["抓挠"]])

    labels = np.array([CODE[CLASSES[i]] for i in proba.argmax(axis=1)])
    legacy = windows_to_events(_majority_smooth(labels, 5), proba.max(axis=1),
                               WIN, STEP, FS, BASE)
    n_legacy = len([e for e in legacy if e["behavior_type"] == CODE["抓挠"]])

    assert n_legacy == 2, f"老路子该报 2 次才有对比意义，实际 {n_legacy}"
    assert n_v2 == 1, f"v2 该合成 1 次，实际 {n_v2}"


def test_v2_does_not_merge_across_a_real_gap():
    """跟上一条成对：断得够久（5 窗 = 5s > event_gap_s=4s）就**不能**合。

    只测"会合并"的话，一个无条件把所有抓挠合成一段的实现也是绿的——
    而那会把一整天的零星抓挠算成一次。
    """
    proba = _probs([(1, 20), (0, 4), (1, 5), (0, 4), (1, 20)])
    v2 = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS)
    n = len([e for e in v2 if e["behavior_type"] == CODE["抓挠"]])
    assert n == 2, f"隔了 5s 不该合并，实际 {n} 段"


def test_v2_drops_a_lone_weak_window():
    """孤零零一个不够强的抓挠窗口要被丢掉，老路子会把它报成一次事件。"""
    _majority_smooth, windows_to_events = _legacy()

    proba = _probs([(1, 20), (0, 1), (1, 20)], conf=0.6)
    v2 = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS)
    assert not [e for e in v2 if e["behavior_type"] == CODE["抓挠"]]

    labels = np.array([CODE[CLASSES[i]] for i in proba.argmax(axis=1)])
    legacy = windows_to_events(_majority_smooth(labels, 5), proba.max(axis=1),
                               WIN, STEP, FS, BASE)
    # 老路子那边平滑会不会吃掉它取决于 k，这里只断言 v2 的行为，
    # 但把老路子的结果打出来，免得以为这条在比空集
    assert isinstance(legacy, list)


def test_unknown_class_is_skipped_not_written_as_zero():
    """模型有这个类别但业务侧没编码时跳过，不能塞个 0 进去——
    塞 0 的话库里会多出一批"未知"事件，看起来像识别失败。"""
    proba = _probs([(1, 20), (0, 6), (1, 20)])
    evs = S.stabilize_events(proba, CLASSES, {"活动": 1}, BASE, WIN, STEP, FS)
    assert evs, "全被跳过了，这条测试在空转"
    assert {e["behavior_type"] for e in evs} == {1}


def test_empty_input_is_empty():
    assert S.stabilize_events(np.empty((0, 3), np.float32), CLASSES, CODE,
                              BASE, WIN, STEP, FS) == []


def test_confidence_is_the_segment_mean_like_before():
    """置信度用段内平均，跟老路子一致。换成 max 的话库里的置信度整体抬高，
    跟历史数据不可比。"""
    proba = _probs([(1, 20), (0, 6), (1, 20)])
    evs = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS)
    sc = [e for e in evs if e["behavior_type"] == CODE["抓挠"]]
    assert len(sc) == 1
    # 干净数据下段内每个窗口都是 0.95，平均也是 0.95
    assert abs(sc[0]["confidence"] - 0.95) < 1e-3


def test_switch_back_to_legacy_is_possible():
    """出问题要能回退。开关读的是 settings.postprocess。"""
    from config import settings
    assert hasattr(settings, "postprocess")
    assert settings.postprocess == "v2", "默认该是 v2"


def test_segment_end_uses_the_window_length_not_the_stride():
    """最后一个窗口的结束时间 = ts + window_s（2s），不是 + stride_s（1s）。

    window_s 和 stride_s 传反了的话，段的**起点全对、终点少 1 秒**，
    duration_sec 跟着错——而每一条记录看起来都完全正常。
    （变异测试里把这两个参数对调，上面那些测试全是绿的。）
    """
    proba = _probs([(1, 10), (0, 4)])
    evs = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS)
    assert len(evs) == 2, evs
    last = evs[-1]
    # 14 个窗口，最后一个起点 = 13 * 1s，结束 = 起点 + window_s(2s) = 15s
    assert (last["end_time"] - BASE) == 15_000, \
        f"末段结束在 {(last['end_time'] - BASE) / 1000}s，应该是 15s"
    # 第一段的边界也钉住：起点 0，结束 = 第 10 个窗口的起点 = 10s
    assert (evs[0]["start_time"] - BASE) == 0
    assert (evs[0]["end_time"] - BASE) == 10_000


def test_confidence_is_the_mean_not_the_max():
    """段内置信度不一样时要取平均。

    老路子写的就是段内平均（windows_to_events 里 conf_sum / conf_cnt）。
    换成 max 的话库里的置信度整体抬高，跟历史数据不可比——
    而且不报错。上面那条干净数据的测试里每个窗口都是 0.95，
    平均和最大相等，**验不到这件事**（变异测试证实了）。
    """
    rows = []
    for _ in range(20):
        p = np.array([0.02, 0.96, 0.02], np.float32)
        rows.append(p)
    # 一段抓挠，段内置信度差很多：0.99 / 0.60 / 0.99 / 0.60
    for v in (0.99, 0.60, 0.99, 0.60):
        p = np.full(3, (1.0 - v) / 2, np.float32)
        p[0] = v
        rows.append(p)
    for _ in range(20):
        rows.append(np.array([0.02, 0.96, 0.02], np.float32))
    proba = np.array(rows, np.float32)

    evs = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS)
    sc = [e for e in evs if e["behavior_type"] == CODE["抓挠"]]
    assert len(sc) == 1, sc
    mean = (0.99 + 0.60 + 0.99 + 0.60) / 4
    assert abs(sc[0]["confidence"] - mean) < 1e-3, \
        f"该是段内平均 {mean:.3f}，实际 {sc[0]['confidence']}（取成最大值了？）"
    assert abs(sc[0]["confidence"] - 0.99) > 0.05, "取成最大值了"


def test_algo_is_viterbi_not_stable():
    """走的必须是 viterbi（稳定版 v2），不是 stable（稳定版一代）。

    两者在抖动数据上差很多——实测同一段输入 viterbi 出 5 段、stable 出 15 段。
    传错了不报错，只是效果退回上一代，而"效果好"正是换这套的理由。
    （干净的合成数据上两者结果一样，所以这条要用**抖的**数据。）
    """
    rng = np.random.default_rng(0)
    n = 120
    z = rng.normal(0, 1.2, (n, 3))
    for i in range(0, n, 17):
        z[i:i + 3, 0] += 3.0
    z[:, 1] += 1.2
    e = np.exp(z - z.max(axis=1, keepdims=True))
    proba = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    v = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS, algo="viterbi")
    st = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS, algo="stable")
    assert v != st, "这段输入区分不了两种算法，这条测试在空转"
    # 默认（不传 algo）必须等于 viterbi
    default = S.stabilize_events(proba, CLASSES, CODE, BASE, WIN, STEP, FS)
    assert default == v, "默认走的不是 viterbi"
