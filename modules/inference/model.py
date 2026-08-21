import json
import logging
from datetime import datetime, timezone as dt_tz
from enum import IntEnum
from pathlib import Path

import joblib
import numpy as np

from config import settings
from modules.inference import features as F
from modules.inference.gravity import prepare_windows

_logger = logging.getLogger(__name__)


class BehaviorLabel(IntEnum):
    UNKNOWN  = 0
    MOVEMENT = 1
    SLEEP    = 2
    SCRATCH  = 3


# imu_train 类别中文名 → BehaviorLabel
# 类别顺序由 ml_rf.json 的 classes 决定，不写死下标——换模型时顺序可能变
_ZH_TO_LABEL: dict[str, int] = {
    "抓挠": BehaviorLabel.SCRATCH,
    "活动": BehaviorLabel.MOVEMENT,
    "睡觉": BehaviorLabel.SLEEP,
}

_LABEL_ZH: dict[int, str] = {
    int(BehaviorLabel.UNKNOWN):  "未知",
    int(BehaviorLabel.MOVEMENT): "活动",
    int(BehaviorLabel.SLEEP):    "睡觉",
    int(BehaviorLabel.SCRATCH):  "抓挠",
}

# classes 缺失时的兜底顺序（与仓库内已提交模型一致）
_DEFAULT_CLASSES = ["抓挠", "活动", "睡觉"]


# ---------------------------------------------------------------------------
# 滑动窗口分段
# ---------------------------------------------------------------------------

def segment(data: np.ndarray, window_samples: int, step_samples: int) -> list[np.ndarray]:
    """对数据进行滑动窗口分段，返回窗口列表。"""
    windows, start = [], 0
    while start + window_samples <= len(data):
        windows.append(data[start: start + window_samples])
        start += step_samples
    return windows


# ---------------------------------------------------------------------------
# 行为事件合并：将连续相同标签的窗口合并为一个事件
# ---------------------------------------------------------------------------

def windows_to_events(
    labels: np.ndarray,
    confidences: np.ndarray,
    window_samples: int,
    step_samples: int,
    fs: int,
    base_ts_ms: int,
) -> list[dict]:
    """
    将逐窗口预测结果合并为行为事件。

    base_ts_ms：批次第一个采样点的 UTC 毫秒时间戳。
    返回字典列表，包含字段：behavior_type、start_time、end_time、confidence
    """
    if len(labels) == 0:
        return []

    events = []
    cur_label = int(labels[0])
    cur_conf_sum = float(confidences[0])
    cur_conf_cnt = 1
    cur_start_sample = 0

    for idx in range(1, len(labels)):
        lbl = int(labels[idx])
        if lbl == cur_label:
            cur_conf_sum += float(confidences[idx])
            cur_conf_cnt += 1
        else:
            end_sample = idx * step_samples + window_samples
            events.append({
                "behavior_type": cur_label,
                "start_time": base_ts_ms + int(cur_start_sample / fs * 1000),
                "end_time":   base_ts_ms + int(end_sample / fs * 1000),
                "confidence": round(cur_conf_sum / cur_conf_cnt, 4),
            })
            cur_label = lbl
            cur_conf_sum = float(confidences[idx])
            cur_conf_cnt = 1
            cur_start_sample = idx * step_samples

    end_sample = (len(labels) - 1) * step_samples + window_samples
    events.append({
        "behavior_type": cur_label,
        "start_time": base_ts_ms + int(cur_start_sample / fs * 1000),
        "end_time":   base_ts_ms + int(end_sample / fs * 1000),
        "confidence": round(cur_conf_sum / cur_conf_cnt, 4),
    })
    return events


# ---------------------------------------------------------------------------
# 逐窗口标签平滑：滑动多数票，去除单窗口随机噪声翻转
# ---------------------------------------------------------------------------

def _majority_smooth(labels: np.ndarray, k: int = 5) -> np.ndarray:
    """k=5 → 每个位置参考前后各 2 帧，孤立的 1~2 窗口翻转被邻居多数覆盖。"""
    if k <= 1 or len(labels) < k:
        return labels
    half = k // 2
    smoothed = labels.copy()
    for i in range(len(labels)):
        window = labels[max(0, i - half): i + half + 1]
        vals, counts = np.unique(window, return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]
    return smoothed


# ---------------------------------------------------------------------------
# 模型封装
# ---------------------------------------------------------------------------

class BehaviorClassifier:
    def __init__(self, model_path: str | None = None):
        path = Path(model_path or settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        self._model = joblib.load(path)

        self._fs   = settings.imu_sample_rate
        self._win  = int(settings.window_seconds * self._fs)
        self._step = int(self._win * (1 - settings.window_overlap))
        self._conf_threshold = settings.confidence_threshold
        self._smooth_k = settings.smooth_window

        meta = self._load_meta(path)
        self._classes = meta.get("classes") or _DEFAULT_CLASSES
        self._label_map = {
            i: int(_ZH_TO_LABEL.get(name, BehaviorLabel.UNKNOWN))
            for i, name in enumerate(self._classes)
        }

        # 特征布局自动识别：拿模型的 n_features_in_ 反推该用哪套特征
        self._feature_mode, self._n_channels = self._detect_feature_layout()

        self._log_startup(path, meta)

    # -- 初始化辅助 ---------------------------------------------------------

    @staticmethod
    def _load_meta(path: Path) -> dict:
        json_path = path.with_suffix(".json")
        if not json_path.exists():
            _logger.info("未找到模型元数据 %s，跳过参数校验", json_path.name)
            return {}
        try:
            return json.loads(json_path.read_text())
        except Exception as e:
            _logger.warning("读取模型元数据失败: %s", e)
            return {}

    def _detect_feature_layout(self) -> tuple[str, int]:
        """
        根据模型期望的特征数，判断该用哪套特征提取：
          v2 + 8 通道（含 pitch/roll）→ 当前 imu_train 训练管线
          v2 + 6 通道                 → 未附加姿态角的 imu_train 模型
          legacy                      → 旧版 78 维模型（仓库内已提交的那个）
        """
        n_expected = getattr(self._model, "n_features_in_", None)
        candidates = [
            ("v2", 8, F.feature_dim(self._win, 8, self._fs)),
            ("v2", 6, F.feature_dim(self._win, 6, self._fs)),
            ("legacy", 6, 78),
        ]
        if n_expected is None:
            _logger.warning("模型未暴露 n_features_in_，默认使用 v2/8通道特征")
            return "v2", 8
        for mode, n_ch, dim in candidates:
            if dim == n_expected:
                return mode, n_ch
        raise ValueError(
            f"无法识别模型特征布局：模型期望 {n_expected} 维，"
            f"已知布局为 " + ", ".join(f"{m}/{c}通道={d}" for m, c, d in candidates)
            + "。请确认 weights/ 下的模型与当前 imu_train 特征提取版本一致。"
        )

    def _log_startup(self, path: Path, meta: dict) -> None:
        infer_stride_s = round(self._step / self._fs, 3)
        n_feat = getattr(self._model, "n_features_in_", "?")
        mode_desc = {
            "v2": f"imu_train 当前版本（{self._n_channels} 通道"
                  f"{'，含 pitch/roll' if self._n_channels >= 8 else ''}）",
            "legacy": "旧版 78 维（建议尽快更换为新模型）",
        }[self._feature_mode]

        _logger.info("=" * 60)
        _logger.info("行为分类器加载完成")
        _logger.info("  模型文件 : %s  (%s)", path, type(self._model).__name__)
        _logger.info("  特征布局 : %s  维度=%s", mode_desc, n_feat)
        _logger.info("  类别顺序 : %s", self._classes)

        t_hz       = meta.get("hz")
        t_window_s = meta.get("window_s")
        t_stride_s = meta.get("stride_s")
        t_ga       = meta.get("gravity_aligned")

        if t_hz:
            _logger.info("  训练参数 : fs=%sHz  window=%ss  stride=%ss  gravity_align=%s",
                         t_hz, t_window_s, t_stride_s, t_ga)
        _logger.info("  推理参数 : fs=%dHz  window=%ss  stride=%ss  gravity_align=True  "
                     "conf_threshold=%.2f  smooth_k=%d",
                     self._fs, settings.window_seconds, infer_stride_s,
                     self._conf_threshold, self._smooth_k)

        warns = []
        fix = "修改 docker-compose.yml 中对应环境变量后执行 docker compose down && docker compose up -d 生效"
        if t_hz and t_hz != self._fs:
            warns.append(f"采样率不一致  训练={t_hz}Hz  推理={self._fs}Hz\n"
                         f"       → 将 IMU_SAMPLE_RATE 改为 {t_hz}，{fix}")
        if t_window_s and abs(t_window_s - settings.window_seconds) > 0.01:
            warns.append(f"窗口长度不一致  训练={t_window_s}s  推理={settings.window_seconds}s\n"
                         f"       → 将 WINDOW_SECONDS 改为 {t_window_s}，{fix}")
        if t_stride_s and abs(t_stride_s - infer_stride_s) > 0.01:
            warns.append(f"步长不一致  训练={t_stride_s}s  推理={infer_stride_s}s\n"
                         f"       → 调整 WINDOW_OVERLAP 使步长={t_stride_s}s，{fix}")
        if t_ga is not None and not bool(t_ga):
            warns.append("训练时未开启重力对齐，但推理已开启\n"
                         "       → 重新训练时建议开启重力对齐，或联系模型维护者")
        if self._feature_mode == "legacy":
            warns.append("当前模型使用旧版 78 维特征，与 imu_train 最新特征管线不一致\n"
                         "       → 用最新 imu_train 重新训练后替换 weights/ml_rf.pkl 与 ml_rf.json")

        if warns:
            for w in warns:
                _logger.warning("  ⚠️  %s", w)
        else:
            _logger.info("  参数校验 : ✓ 训练与推理参数一致")
        _logger.info("=" * 60)

    # -- 推理 ---------------------------------------------------------------

    def extract(self, windows: np.ndarray) -> np.ndarray:
        """(N, T, 6) 原始窗口 → (N, n_features)，预处理顺序与训练侧一致。"""
        if self._feature_mode == "legacy":
            aligned = prepare_windows(windows, use_gravity_align=True, with_tilt=False)
            return np.stack([F.extract_one_legacy(w, self._fs) for w in aligned])
        prepared = prepare_windows(
            windows, use_gravity_align=True, with_tilt=self._n_channels >= 8
        )
        return F.extract_features(prepared, self._fs)

    def predict_windows(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        data: (N, 6) 原始 IMU 序列
        返回 (labels, confidences)，均为逐窗口结果，labels 已映射为 BehaviorLabel。
        """
        windows = segment(data, self._win, self._step)
        if not windows:
            return np.empty(0, dtype=int), np.empty(0)

        X = self.extract(np.stack(windows))
        raw_labels = self._model.predict(X)
        labels = np.array([self._label_map.get(int(l), int(BehaviorLabel.UNKNOWN))
                           for l in raw_labels])

        if hasattr(self._model, "predict_proba"):
            confidences = self._model.predict_proba(X).max(axis=1)
        else:
            confidences = np.ones(len(labels))

        if self._conf_threshold > 0:
            labels = np.where(confidences >= self._conf_threshold,
                              labels, int(BehaviorLabel.UNKNOWN))

        labels = _majority_smooth(labels, k=self._smooth_k)
        return labels, confidences

    def predict(
        self,
        data: np.ndarray,
        base_ts_ms: int,
        device_id: int | None = None,
    ) -> list[dict]:
        """
        data       : (N, 6) float32，按时间顺序排列的 IMU 采样数据
        base_ts_ms : data[0] 对应的 UTC 毫秒时间戳

        返回行为事件列表（参见 windows_to_events）。
        """
        labels, confidences = self.predict_windows(data)
        if len(labels) == 0:
            return []

        if settings.verbose_inference:
            self._log_windows(labels, confidences, base_ts_ms, device_id)

        return windows_to_events(
            labels, confidences, self._win, self._step, self._fs, base_ts_ms
        )

    def _log_windows(self, labels, confidences, base_ts_ms, device_id) -> None:
        pc_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step_ms = self._step * 1000 // self._fs
        dev_tag = f"设备{device_id} " if device_id is not None else ""
        for i, (lbl, conf) in enumerate(zip(labels, confidences)):
            chip_time = datetime.fromtimestamp(
                (base_ts_ms + i * step_ms) / 1000, tz=dt_tz.utc
            ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            _logger.info(
                "[PC %s | 片上 %s]  %sML=%s(%d%%)",
                pc_now, chip_time, dev_tag,
                _LABEL_ZH.get(int(lbl), "未知"), int(conf * 100),
            )


# ---------------------------------------------------------------------------
# 兼容旧调用方（train/train.py、tests/test_1_inference.py）
# 合成数据时代的遗留脚本，与线上推理管线已无关联；线上请用 BehaviorClassifier.extract
# ---------------------------------------------------------------------------

_time_features = F._legacy_time_features
_freq_features = F._legacy_freq_features


def extract_features(window: np.ndarray, fs: int) -> np.ndarray:
    """[已废弃] 旧版 78 维单窗口特征，仅供遗留脚本使用。"""
    return F.extract_one_legacy(window, fs)


# 单例模式，启动时加载一次
_classifier: BehaviorClassifier | None = None


def get_classifier() -> BehaviorClassifier:
    """获取全局分类器单例，首次调用时加载模型。"""
    global _classifier
    if _classifier is None:
        _classifier = BehaviorClassifier()
    return _classifier
