"""重采样 + 几何来源 + 标签映射。

这三件事的共同点：错了都**不报错**，只是结果变差或者少几条记录。
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from modules.inference.geometry import needs_resample, resolve as _resolve
from modules.inference.labels import LABEL_ZH, ZH_TO_LABEL, BehaviorLabel
from modules.inference.resample import resample_training_match


# ── 重采样 ────────────────────────────────────────────────────────────────


def test_vendored_resampler_is_byte_identical_to_imu_train():
    """副本不能跟 imu_train 那份分家。

    换成别的重采样算法跟训练不一致（跟 scipy.resample_poly 实测差 6~8%），
    效果会掉，而且查不到这里。
    """
    src = os.path.expanduser(
        os.environ.get("IMU_TRAIN", "~/imu_train") + "/src/data/resample_training_match.py")
    if not os.path.exists(src):
        pytest.skip(f"找不到 {src}")
    with open(src, encoding="utf-8") as f:
        want = f.read()
    with open(os.path.join(ROOT, "modules", "inference", "resample.py"),
              encoding="utf-8") as f:
        got = f.read()
    marker = "# ============================================================================\n\n"
    i = got.rfind(marker)
    assert i >= 0, "副本头部说明被删了"
    assert got[i + len(marker):] == want, "副本跟 imu_train 那份不一样了"


def test_resample_25_to_16_gives_the_right_length():
    x = np.zeros((250, 6), np.float32)          # 10 秒 @25Hz
    y = resample_training_match(x, 25, 16)
    assert y.shape == (160, 6), f"10 秒 @16Hz 该是 160 点，实际 {y.shape}"


def test_resampler_has_no_identity_shortcut():
    """**这个函数 source==target 时会掉一个点**，所以调用方必须先判相等。

    钉住这个事实：哪天它自己加了短路，这条会挂，那时候可以去掉调用处的判断。
    不钉的话，调用处那个 if 看起来像多余的，容易被"顺手清理"掉。
    """
    x = np.zeros((250, 6), np.float32)
    assert resample_training_match(x, 16, 16).shape == (249, 6)


# ── 几何来自模型元数据 ────────────────────────────────────────────────────


def test_geometry_comes_from_meta_not_env():
    """窗口/步长/采样率按模型自己的元数据，环境变量只兜底。

    以前是反过来的：环境变量说了算，模型元数据只用来打个告警。
    结果仓库里那个 16Hz 模型一直在吃 25Hz、50 点的窗口——特征维度
    （193）还正好一样，什么都没报。
    """
    g = _resolve({"hz": 16, "window_s": 1.0, "stride_s": 0.5,
                  "window_size": 16, "stride": 8},
                 device_hz=25, fallback_window_s=2.0, fallback_overlap=0.5)
    assert (g["fs"], g["win"], g["step"]) == (16, 16, 8)
    assert g["device_fs"] == 25 and g["need_resample"] is True


def test_geometry_falls_back_to_env_when_meta_is_empty():
    """元数据缺项时才用环境变量——老模型没有这些字段也得能跑。"""
    g = _resolve({}, device_hz=25, fallback_window_s=2.0, fallback_overlap=0.5)
    assert (g["fs"], g["win"], g["step"]) == (25, 50, 25)
    assert g["need_resample"] is False


def test_geometry_derives_points_when_only_seconds_are_given():
    """只给了秒数没给点数时，按模型自己的采样率换算，**不是按设备的**。

    按设备采样率换算的话，窗口点数对了、覆盖的真实时长错了。
    """
    g = _resolve({"hz": 16, "window_s": 2.0, "stride_s": 1.0},
                 device_hz=25, fallback_window_s=2.0, fallback_overlap=0.5)
    assert (g["win"], g["step"]) == (32, 16), "该按 16Hz 换算，不是 25Hz"


def test_model_code_reads_geometry_from_meta():
    """源码层面钉住：__init__ 里读的是 meta，不是只读 settings。

    这条防的是"改回去"——改回去不会有任何测试变红，因为几何错了
    仍然跑得通，只是结果差。
    """
    import ast
    with open(os.path.join(ROOT, "modules", "inference", "model.py"),
              encoding="utf-8") as f:
        src = f.read()
    seg = src[src.index("g = _geom.resolve"):src.index("self._conf_threshold")]
    assert "meta" in seg, "几何没把 meta 传进去"
    for k in ("_fs", "_win", "_step", "_need_resample"):
        assert f'g["{k.lstrip("_")}"]' in seg or k in seg, f"{k} 没从几何模块取"
    ast.parse(src)


def test_needs_resample_is_false_when_rates_match():
    """采样率一致时**不能**重采样。

    写死成"总是重采样"的话，每批数据白掉一个点（那个函数没有恒等短路），
    不报错、也没有任何迹象。源码检查拦不住这个——变异测试里把它改成
    恒 True，那条源码检查照样绿。
    """
    from modules.inference.geometry import needs_resample, resolve as _resolve
    assert needs_resample(25, 16) is True
    assert needs_resample(16, 16) is False
    assert needs_resample("16", 16) is False, "字符串也要当成数比"
    # 取不到就别动数据，不要按瞎猜的频率重采样
    assert needs_resample(None, 16) is False


def test_resample_is_guarded_by_an_equality_check():
    """_prepare 里必须先判 source != target 再重采样。

    不判的话每批数据白掉一个点，而且是在采样率本来就一致的情况下——
    最没必要的一种损失。
    """
    with open(os.path.join(ROOT, "modules", "inference", "model.py"),
              encoding="utf-8") as f:
        src = f.read()
    seg = src[src.index("def _prepare"):src.index("def predict_proba_windows")]
    assert "self._need_resample" in seg, "没判就直接重采样了"


# ── 标签映射 ──────────────────────────────────────────────────────────────


def test_shake_is_folded_into_movement():
    """甩身体并进活动（业务上不单独统计它）。"""
    assert ZH_TO_LABEL["甩身体"] == int(BehaviorLabel.MOVEMENT)


def test_not_worn_has_its_own_code():
    """未佩戴单独一个编码，**不并进"未知"**。

    并进去的话"不知道"和"项圈没戴"就分不开了，而后者是能确定的事实。
    也不能并进睡觉——以前没这个类别时就是被判成睡觉，sleep_min 虚高。
    """
    assert ZH_TO_LABEL["未佩戴"] == int(BehaviorLabel.NOT_WORN)
    assert int(BehaviorLabel.NOT_WORN) not in (
        int(BehaviorLabel.UNKNOWN), int(BehaviorLabel.SLEEP),
        int(BehaviorLabel.MOVEMENT), int(BehaviorLabel.SCRATCH))
    assert LABEL_ZH[int(BehaviorLabel.NOT_WORN)] == "未佩戴"


def test_existing_codes_did_not_move():
    """老编码一个都不能变——库里已经有按这些值存的历史数据。"""
    assert (int(BehaviorLabel.UNKNOWN), int(BehaviorLabel.MOVEMENT),
            int(BehaviorLabel.SLEEP), int(BehaviorLabel.SCRATCH)) == (0, 1, 2, 3)


def test_jobs_does_not_keep_its_own_copy_of_the_table():
    """写库那边用的必须是同一张表。

    抄一份的话，加类别时漏改那处：behavior 是对的、behavior_label 写成
    "未知"——两列自相矛盾，而没有任何报错。
    """
    import ast
    with open(os.path.join(ROOT, "scheduler", "jobs.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_BEHAVIOR_ZH":
                    assert isinstance(node.value, ast.Name), \
                        "_BEHAVIOR_ZH 又自己写了一份字面量，应该指向 labels.LABEL_ZH"
                    assert node.value.id == "LABEL_ZH"
                    return
    pytest.fail("找不到 _BEHAVIOR_ZH")


def test_shipped_default_model_is_the_five_class_one():
    """默认模型要是 5 类那个；老的 3 类模型也要还在。"""
    import json

    from config import settings
    assert "stable_v2_rf" in settings.model_path, settings.model_path
    p = os.path.join(ROOT, settings.model_path)
    assert os.path.exists(p), f"默认模型文件不在：{p}"
    with open(os.path.splitext(p)[0] + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["classes"] == ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]
    assert all(c in ZH_TO_LABEL for c in meta["classes"])
    # 老模型保留
    assert os.path.exists(os.path.join(ROOT, "weights", "ml_rf.pkl")), \
        "老的 3 类模型被删了——说好要保留的"


# ── 整条链：5 类模型的几何 + 映射 ─────────────────────────────────────────

_CLASSES5 = ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]
_FS, _WIN, _STEP = 16, 16, 8      # 模型自己的几何：1.0s 窗口、0.5s 步长
_BASE = 1_757_000_000_000


def _p5(spec, conf=0.95):
    rows = []
    for c, k in spec:
        for _ in range(k):
            p = np.full(5, (1.0 - conf) / 4, np.float32)
            p[c] = conf
            rows.append(p)
    return np.array(rows, np.float32)


def _ev(spec, conf=0.95):
    from modules.inference import stabilize as S
    return S.stabilize_events(_p5(spec, conf), _CLASSES5, ZH_TO_LABEL,
                              _BASE, _WIN, _STEP, _FS)


def test_shake_between_two_movements_becomes_one_event():
    """活动 → 甩身体 → 活动：两边映射成同一个编码，要合成**一段**。

    不合并的话库里是三条紧挨着、编码相同的记录——时长加起来没错，
    但"活动了几次"偏大，而看记录完全看不出为什么会断开。
    """
    evs = _ev([(0, 20), (4, 6), (0, 20)])
    assert len(evs) == 1, [(LABEL_ZH[e["behavior_type"]], e["start_time"]) for e in evs]
    assert evs[0]["behavior_type"] == int(BehaviorLabel.MOVEMENT)


def test_not_worn_stays_its_own_event():
    """未佩戴不并进任何东西——它是独立的一段。"""
    evs = _ev([(0, 20), (3, 30), (0, 20)])
    codes = [e["behavior_type"] for e in evs]
    assert int(BehaviorLabel.NOT_WORN) in codes, \
        [(LABEL_ZH[c]) for c in codes]
    nw = [e for e in evs if e["behavior_type"] == int(BehaviorLabel.NOT_WORN)]
    assert len(nw) == 1 and nw[0]["end_time"] > nw[0]["start_time"]


def test_scratch_still_absorbs_shake_after_the_remap():
    """**映射不能影响后处理**。

    抓挠吞并前后甩身体那条规则是在后处理层按"甩身体"这个**类别**做的；
    如果在更早的地方就把甩身体改名成活动，这条规则会少一条输入，
    抓挠段短一截——而段还在、类别还对，只是短了，最难发现的那种。
    """
    evs = _ev([(0, 20), (4, 2), (2, 6), (4, 2), (0, 20)])
    sc = [e for e in evs if e["behavior_type"] == int(BehaviorLabel.SCRATCH)]
    assert len(sc) == 1, evs
    dur = sc[0]["end_time"] - sc[0]["start_time"]
    # 抓挠 6 窗（3s）+ 前后各 2 窗甩身体（各 1s）= 5s
    assert dur == 5000, f"该吞并前后的甩身体（3s+1s+1s=5s），实际 {dur / 1000}s"
