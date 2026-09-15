#!/usr/bin/env python3
"""把 imu_train 训练出来的 RF 模型装进 weights/，**装之前全部检查一遍**。

    python3 scripts/import_model.py ~/imu_train/results_xxx/16hz_.../rf/ml_rf.pkl

为什么不是 `cp`：模型换了之后，一堆东西会静默地对不上，而每一样都不报错，
只是结果变差或者变错：

  · 采样率：模型按 16Hz 训练，服务按 IMU_SAMPLE_RATE=25 推理。32 个点在
    25Hz 下是 1.28 秒，不是训练时的 2 秒，特征整体偏。
  · 窗口/步长：片段时间戳整体错位，而每一段看着都正常。
  · 类别顺序：predict_proba 的列序跟 classes 对不上，概率安到别的类别上。
  · 新类别：模型多了"未佩戴"，而 BehaviorLabel 里没有这个编码——那些窗口
    会被悄悄丢掉，看起来像"这段没识别出东西"。
  · 特征维度：模型期望的维数跟当前特征提取版本对不上，启动时才炸。

所以这个脚本**先验后装**，有问题就拒绝，并且明确说要改哪个环境变量。
--force 可以强行装（知道自己在干什么时），但该说的照样说。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

WEIGHTS = os.path.join(ROOT, "weights")


def meta_path_of(pkl: str) -> str:
    """跟 .pkl 同名的 .json。imu_train 的训练脚本就是这么成对产出的。"""
    return os.path.splitext(pkl)[0] + ".json"


def load_meta(pkl: str) -> dict:
    mp = meta_path_of(pkl)
    if not os.path.exists(mp):
        sys.exit(f"找不到元数据 {mp}。\n"
                 "  .pkl 和同名 .json 是成对的，只拷 .pkl 的话类别顺序、\n"
                 "  窗口长度、采样率全部无从得知——服务会按默认值跑，而那多半是错的。")
    with open(mp, encoding="utf-8") as f:
        return json.load(f)


def check(meta: dict, settings) -> tuple[list[str], list[str]]:
    """返回 (拦下来的问题, 只是提醒)。"""
    bad: list[str] = []
    warn: list[str] = []

    classes = meta.get("classes")
    if not classes:
        bad.append("元数据里没有 classes，类别顺序无从得知")
        classes = []

    hz = meta.get("hz")
    if hz and int(hz) != int(settings.imu_sample_rate):
        # 现在有重采样了（modules/inference/resample.py，跟训练同一套算法），
        # 所以这不再是拦截项——但要说出来，好让人知道这条路径被用上了
        warn.append(
            f"模型是 {hz}Hz 训练的，设备上报 {settings.imu_sample_rate}Hz "
            f"（IMU_SAMPLE_RATE）→ 推理时会重采样到 {hz}Hz"
            f"（training_match，跟训练同一套算法）。\n"
            f"    **不要**为此去改 IMU_SAMPLE_RATE：那个值必须是设备真实的\n"
            f"    上报速率，改了会让重采样按错的源频率算，反而全错。")

    # 窗口长度和步长现在**以模型元数据为准**（见 model.py 里
    # BehaviorClassifier.__init__），环境变量只在元数据缺这几项时兜底。
    # 所以它们对不上不再是问题，但同样要说出来
    ws = meta.get("window_s")
    if ws and abs(float(ws) - float(settings.window_seconds)) > 0.01:
        warn.append(f"窗口长度按模型元数据走：{ws}s（WINDOW_SECONDS={settings.window_seconds} "
                    f"只在元数据缺这项时兜底）")
    ss = meta.get("stride_s")
    if ss:
        infer_stride = float(settings.window_seconds) * (1.0 - float(settings.window_overlap))
        if abs(float(ss) - infer_stride) > 0.01:
            warn.append(f"步长按模型元数据走：{ss}s（配置算出来是 "
                        f"{round(infer_stride, 3)}s，只在元数据缺这项时兜底）")

    ga = meta.get("gravity_aligned")
    if ga is not None and not bool(ga):
        warn.append("训练时没开重力对齐，而服务这边是开着的。确认一下。")

    # 新类别：BehaviorLabel 里没有编码的，识别出来也写不进库
    from modules.inference.labels import ZH_TO_LABEL as _ZH_TO_LABEL
    unknown = [c for c in classes if c not in _ZH_TO_LABEL]
    if unknown:
        bad.append(
            f"模型有这些类别，但 BehaviorLabel 里没有对应编码：{unknown}。\n"
            f"    这些窗口识别出来也**写不进库**，会被静默丢掉——\n"
            f"    表现是「这段没识别出东西」，而不是报错。\n"
            f"    → 在 modules/inference/model.py 的 BehaviorLabel 和 _ZH_TO_LABEL、\n"
            f"      _LABEL_ZH 里补上，写库那边的 _BEHAVIOR_ZH 也要补。")

    return bad, warn


def feature_check(pkl: str, meta: dict, settings) -> list[str]:
    """模型期望的特征维数能不能对上当前的特征提取。

    这一条要真的把 .pkl 读起来，所以依赖装不全时跳过——
    **跳过要说出来**，不能让人以为检查通过了。
    """
    try:
        import joblib

        from modules.inference import features as F
    except ImportError as e:
        print(f"  ⚠ 跳过特征维度检查（{e}）。装依赖的机器上再跑一次。")
        return []
    try:
        model = joblib.load(pkl)
    except Exception as e:  # noqa: BLE001
        return [f"读不了 {pkl}：{e}"]
    n = getattr(model, "n_features_in_", None)
    if n is None:
        return []
    win = int(float(meta.get("window_s") or settings.window_seconds)
              * int(meta.get("hz") or settings.imu_sample_rate))
    fs = int(meta.get("hz") or settings.imu_sample_rate)
    cands = [("v2/8通道", F.feature_dim(win, 8, fs)),
             ("v2/6通道", F.feature_dim(win, 6, fs)),
             ("legacy/6通道", 78)]
    if any(d == n for _, d in cands):
        hit = next(name for name, d in cands if d == n)
        print(f"  特征布局：{hit}（{n} 维）")
        return []
    return [f"特征维度对不上：模型期望 {n} 维，已知布局为 "
            + "、".join(f"{nm}={d}" for nm, d in cands)
            + "。\n    当前特征提取版本跟训练时不是一套。"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", help="imu_train 训练产出的 .pkl（同名 .json 必须在旁边）")
    ap.add_argument("--force", action="store_true",
                    help="检查没过也装。该说的照样说")
    ap.add_argument("--dry-run", action="store_true", help="只检查，不动文件")
    args = ap.parse_args()

    pkl = os.path.abspath(os.path.expanduser(args.pkl))
    if not os.path.exists(pkl):
        sys.exit(f"找不到 {pkl}")

    from config import settings

    meta = load_meta(pkl)
    print(f"源模型：{pkl}")
    print(f"  类别  ：{meta.get('classes')}")
    print(f"  几何  ：{meta.get('hz')}Hz，窗口 {meta.get('window_s')}s，"
          f"步长 {meta.get('stride_s')}s，重力对齐={meta.get('gravity_aligned')}")
    if meta.get("macro_f1") is not None:
        print(f"  指标  ：accuracy={meta.get('accuracy')}  macro_f1={meta.get('macro_f1')}")

    bad, warn = check(meta, settings)
    bad += feature_check(pkl, meta, settings)

    for w in warn:
        print(f"  ⚠ {w}")
    if bad:
        print("\n检查没过：")
        for b in bad:
            print(f"  ✗ {b}")
        if not args.force:
            print("\n没有装。修好上面的问题，或者 --force 强行装。")
            return 1
        print("\n--force：照样装。上面那些问题还在。")

    if args.dry_run:
        print("\n--dry-run，没动文件。")
        return 0

    os.makedirs(WEIGHTS, exist_ok=True)
    for src, dst in ((pkl, os.path.join(WEIGHTS, "ml_rf.pkl")),
                     (meta_path_of(pkl), os.path.join(WEIGHTS, "ml_rf.json"))):
        shutil.copy2(src, dst)
        print(f"  → {dst}（{os.path.getsize(dst)} 字节）")

    print("\n装好了。接下来：")
    print("  git add weights/ && git commit && git push")
    print("  docker compose down && docker compose up -d --build")
    print("  docker logs algo_service --tail 50   # 启动日志里会再核对一遍几何")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
