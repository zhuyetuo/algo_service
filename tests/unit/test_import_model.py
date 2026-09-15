"""装模型脚本：**先验后装**。

这些检查存在的理由是：模型换了之后一堆东西会静默对不上，每一样都不报错，
只是结果变差或者变错。所以每条测试都钉住"这个问题必须被拦下来"。

不依赖 sklearn/joblib——最该被验的那条（新类别写不进库）恰恰在装不上
依赖的机器上也要能跑。
"""

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(ROOT))


def _script():
    p = os.path.join(ROOT, "scripts", "import_model.py")
    spec = importlib.util.spec_from_file_location("_import_model", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _S:
    """假的 settings，按当前部署的默认值。"""
    imu_sample_rate = 25
    window_seconds = 2.0
    window_overlap = 0.5


def _meta(**kw):
    d = {"hz": 25, "classes": ["活动", "睡觉", "抓挠"], "window_size": 50,
         "window_s": 2.0, "stride_s": 1.0, "gravity_aligned": True}
    d.update(kw)
    return d


def test_matching_model_passes():
    """全都对得上时不该有意见——不然这些检查就成了摆设，
    谁都会加 --force，等于没有检查。"""
    bad, warn = _script().check(_meta(), _S())
    assert bad == [], bad
    assert warn == []


def test_sample_rate_mismatch_says_resample_not_relabel():
    """采样率对不上要拦下来，而且**给的办法必须是对的**。

    我第一版写的是"把 IMU_SAMPLE_RATE 改成 16"——那是错的，而且更糟：
    数据本身还是 25Hz，改配置只是骗代码说它是 16Hz，窗口变成 32 点、
    真实时长掉到 1.28 秒，FFT 还会按错的 fs 算。
    正确的做法是**加重采样**（25→16Hz，跟训练同一套算法）。

    窗口时长本身是对的：代码用 window_seconds × fs 算点数，
    2.0s × 25 = 50 点，跟训练的 2.0s × 16 = 32 点**时长一样**，
    差的是密度。这一点也要说清楚，不然人会去改 WINDOW_SECONDS。
    """
    bad, _ = _script().check(_meta(hz=16, window_size=32), _S())
    msg = [b for b in bad if "采样率" in b]
    assert msg, bad
    m = msg[0]
    assert "要加重采样" in m, "没说该怎么办"
    assert "不是改 IMU_SAMPLE_RATE" in m, \
        "还在建议改 IMU_SAMPLE_RATE——那是错的办法，会让效果更差"
    assert "时长**是对的" in m or "时长" in m, "没说清窗口时长其实是对的，人会去改 WINDOW_SECONDS"
    assert "32 点" in m and "50 点" in m, "没把两边的点数摆出来"


def test_window_and_stride_mismatch_are_blocked():
    s = _script()
    bad, _ = s.check(_meta(window_s=4.0), _S())
    assert any("WINDOW_SECONDS 改成 4.0" in b for b in bad), bad
    bad, _ = s.check(_meta(stride_s=0.5), _S())
    assert any("WINDOW_OVERLAP 改成 0.75" in b for b in bad), bad


def test_new_classes_without_a_code_are_blocked():
    """**这条最要紧**：模型多了「未佩戴」「甩身体」，而 BehaviorLabel 里
    没有编码——那些窗口识别出来也写不进库，被静默丢掉，
    表现是「这段没识别出东西」，不是报错。
    """
    bad, _ = _script().check(
        _meta(classes=["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]), _S())
    msg = [b for b in bad if "BehaviorLabel" in b]
    assert msg, bad
    assert "未佩戴" in msg[0] and "甩身体" in msg[0]
    # 已有的三个不能被误报
    assert "活动" not in msg[0].split("：")[1].split("。")[0]


def test_missing_classes_is_blocked():
    bad, _ = _script().check(_meta(classes=None), _S())
    assert any("classes" in b for b in bad)


def test_gravity_align_mismatch_is_only_a_warning():
    """重力对齐对不上是提醒不是拦截：它可能是有意为之，
    而拦下来会让人直接上 --force，把真正该拦的也一起放过去。"""
    bad, warn = _script().check(_meta(gravity_aligned=False), _S())
    assert bad == []
    assert warn and "重力对齐" in warn[0]


def test_missing_meta_json_exits(tmp_path):
    """只拷 .pkl 不拷 .json 要当场停。

    没有 .json 的话类别顺序、窗口长度、采样率全部无从得知，
    服务会按默认值跑——而那多半是错的，且不报错。
    """
    pkl = tmp_path / "m.pkl"
    pkl.write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        _script().load_meta(str(pkl))
    assert "元数据" in str(e.value)


def test_meta_is_the_sibling_json():
    s = _script()
    assert s.meta_path_of("/a/b/ml_rf.pkl") == "/a/b/ml_rf.json"


# ── 标签表只有一份 ────────────────────────────────────────────────────────


def test_label_table_has_no_heavy_imports():
    """labels.py 不能 import joblib/sklearn/numpy。

    装模型的脚本要查这张表；从 model.py 拿的话会连带把推理依赖全拖起来，
    而装不上依赖的机器上那个检查就跳过了——跳过的正是最要紧的那条。
    """
    import ast
    p = os.path.join(ROOT, "modules", "inference", "labels.py")
    with open(p, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    # 看**真的 import 了什么**，不是全文搜关键词——文档字符串里提到
    # "不 import numpy" 会让全文搜索误报（第一版就是这么挂的）
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    heavy = {"joblib", "sklearn", "numpy", "scipy", "pandas"} & mods
    assert not heavy, f"labels.py import 了重依赖：{heavy}"


def test_model_still_exports_the_old_names():
    """老的 import 路径不能断——jobs.py 从 model.py 拿 BehaviorLabel。"""
    import ast
    p = os.path.join(ROOT, "modules", "inference", "model.py")
    with open(p, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "modules.inference.labels":
            for a in node.names:
                names.add(a.asname or a.name)
    assert {"BehaviorLabel", "_ZH_TO_LABEL", "_LABEL_ZH", "_DEFAULT_CLASSES"} <= names, names


def test_label_table_covers_exactly_the_shipped_model():
    """仓库里这个模型的每个类别都要有编码。

    对不上的话服务照常启动、照常识别，只是有些行为写不进库。
    """
    from modules.inference.labels import ZH_TO_LABEL
    with open(os.path.join(ROOT, "weights", "ml_rf.json"), encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    missing = [c for c in classes if c not in ZH_TO_LABEL]
    assert not missing, f"weights/ 里的模型有这些类别却没有编码：{missing}"
