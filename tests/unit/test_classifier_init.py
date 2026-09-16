"""真的把 BehaviorClassifier **构造出来**。

在这条测试之前，整个仓库**没有任何一个测试 new 过它**——于是
`__init__` 里 `meta` 被先用后赋值（UnboundLocalError）这种错能一路合进主干，
而表现是**服务根本起不来**。单测全绿，因为谁都没构造过它。

所以这里盯的不是某个算法细节，是"构造得出来"本身。加字段、调顺序、
挪一行初始化，都先在这里炸，而不是在线上起服务的时候。

没装 sklearn / joblib 的机器上也要能跑（离线装不上），所以打桩——
但**构造的是真的 BehaviorClassifier**，桩只负责顶替"从磁盘 load 出一个估计器"。
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CLASSES = ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]


class _FakeForest:
    """够 BehaviorClassifier 用的最小估计器。"""

    def __init__(self, n_features=193, n_classes=5):
        self.n_features_in_ = n_features
        self.n_classes_ = n_classes

    def predict_proba(self, X):
        return np.full((len(X), self.n_classes_), 1.0 / self.n_classes_, np.float32)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


@pytest.fixture(scope="module")
def model_mod():
    if "joblib" not in sys.modules:
        try:
            import joblib  # noqa: F401
        except ImportError:
            m = types.ModuleType("joblib")
            m.load = lambda *a, **k: _FakeForest()
            m.dump = lambda *a, **k: None
            sys.modules["joblib"] = m
    from modules.inference import model as mm
    return mm


def _write(tmp_path, meta, name="ml_rf"):
    pkl = tmp_path / f"{name}.pkl"
    pkl.write_bytes(b"not-a-real-pickle")      # joblib.load 被打了桩，内容无所谓
    (tmp_path / f"{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return pkl


def _meta(**over):
    m = {"classes": CLASSES, "hz": 16, "window_size": 16, "stride": 8,
         "window_s": 1.0, "stride_s": 0.5, "gravity_aligned": True,
         "label_mode": "majority"}
    m.update(over)
    return m


def _build(model_mod, monkeypatch, pkl, forest):
    monkeypatch.setattr(model_mod.joblib, "load", lambda *a, **k: forest)
    return model_mod.BehaviorClassifier(str(pkl))


# ── 构造得出来 ────────────────────────────────────────────────────────────


def test_classifier_constructs(model_mod, tmp_path, monkeypatch):
    """就是这一条。之前 __init__ 里 meta 先用后赋值，这里会 UnboundLocalError。"""
    clf = _build(model_mod, monkeypatch, _write(tmp_path, _meta()), _FakeForest())
    assert clf._classes == CLASSES


def test_geometry_comes_from_the_model_meta(model_mod, tmp_path, monkeypatch):
    """几何以模型 meta 为准，不是环境变量。

    这条顺带钉住 meta 必须在几何解析**之前**读出来——反过来的话
    要么炸，要么（更糟）读到空 meta 而悄悄用环境变量的几何。
    """
    clf = _build(model_mod, monkeypatch,
                 _write(tmp_path, _meta(hz=16, window_size=16, stride=8)),
                 _FakeForest())
    assert clf._fs == 16
    assert clf._win == 16
    assert clf._step == 8


# ── 只用加速计的模型（feature_select） ────────────────────────────────────


def _acc3_meta():
    return _meta(n_channels=5, n_features=113, feature_set="acc3_gyro_free_113",
                 feature_select={"from_dim": 193, "indices": list(range(113))})


def test_acc3_model_loads_and_records_the_column_selection(model_mod, tmp_path, monkeypatch):
    """113 维的模型要挂得上。

    没有 feature_select 这条路的话，_detect_feature_layout 会拿 113 去对
    193/171/78 三条，一条都不中，报"无法识别特征布局"——而真实情况是
    "这个模型只用其中 113 列"。
    """
    clf = _build(model_mod, monkeypatch, _write(tmp_path, _acc3_meta()),
                 _FakeForest(n_features=113))
    assert clf._feature_select is not None
    assert clf._select_from == 193
    assert clf._feature_mode == "v2" and clf._n_channels == 8, \
        "取列归取列，特征还是照 8 通道算的"


def test_acc3_select_takes_the_columns(model_mod, tmp_path, monkeypatch):
    clf = _build(model_mod, monkeypatch, _write(tmp_path, _acc3_meta()),
                 _FakeForest(n_features=113))
    X = np.arange(2 * 193, dtype=np.float32).reshape(2, 193)
    out = clf._select(X)
    assert out.shape == (2, 113)
    assert np.array_equal(out, X[:, :113])


def test_select_rejects_a_different_source_dim(model_mod, tmp_path, monkeypatch):
    """来的不是 193 维就当场报错。

    按老下标取列**不会报错**——每一维都对到别的特征上，模型照样给得出概率。
    """
    clf = _build(model_mod, monkeypatch, _write(tmp_path, _acc3_meta()),
                 _FakeForest(n_features=113))
    with pytest.raises(ValueError) as e:
        clf._select(np.zeros((2, 171), np.float32))
    assert "193" in str(e.value) and "171" in str(e.value)


def test_pkl_and_json_from_different_runs_are_rejected(model_mod, tmp_path, monkeypatch):
    """json 说取 113 列，pkl 却要 95 维 —— 两者不是一次训练出来的。

    放过去的话取完列喂进去，sklearn 才报维度错，而那时候错误信息
    指向的是特征提取。
    """
    with pytest.raises(ValueError) as e:
        _build(model_mod, monkeypatch, _write(tmp_path, _acc3_meta()),
               _FakeForest(n_features=95))
    assert "不是一次训练出来的" in str(e.value)


def test_out_of_range_indices_are_rejected(model_mod, tmp_path, monkeypatch):
    meta = _meta(feature_select={"from_dim": 193, "indices": [0, 1, 200]})
    with pytest.raises(ValueError) as e:
        _build(model_mod, monkeypatch, _write(tmp_path, meta),
               _FakeForest(n_features=3))
    assert "越界" in str(e.value)


# ── 老模型行为不变 ────────────────────────────────────────────────────────


def test_model_without_feature_select_is_untouched(model_mod, tmp_path, monkeypatch):
    """没有这个字段的老模型，整份 193 维喂进去，一点没变。"""
    clf = _build(model_mod, monkeypatch, _write(tmp_path, _meta()), _FakeForest())
    assert clf._feature_select is None
    X = np.zeros((3, 193), np.float32)
    assert clf._select(X) is X
