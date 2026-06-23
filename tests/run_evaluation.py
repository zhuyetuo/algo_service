"""
一键评估脚本：训练模型、评估准确率、更新 tests/evaluation/ 目录数据文件、运行单元测试。

用法（在容器内运行）：
    python tests/run_evaluation.py            # 使用缓存数据（如有）
    python tests/run_evaluation.py --fresh    # 强制重新生成合成数据后评估
    python tests/run_evaluation.py --no-unit  # 跳过单元测试（仅模型评估）

输出：
    tests/evaluation/model/scenarios_summary.csv
    tests/evaluation/model/classification_report.csv
    tests/evaluation/model/confusion_matrix.csv
    tests/evaluation/model/feature_importance.csv
    tests/evaluation/service/unit_test_results.csv  (--no-unit 时跳过)
"""

import sys
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────

TESTS_DIR     = Path(__file__).parent
EVAL_DIR      = TESTS_DIR / "evaluation"
EVAL_MODEL    = EVAL_DIR / "model"
EVAL_SERVICE  = EVAL_DIR / "service"

EVAL_MODEL.mkdir(parents=True, exist_ok=True)
EVAL_SERVICE.mkdir(parents=True, exist_ok=True)


# ── Feature description table ─────────────────────────────────────────────────

def _feature_descriptions() -> dict[str, str]:
    """Map feat_NNN → Chinese description."""
    time_stats = ["均值", "标准差", "最小值", "最大值", "极差", "均方根", "过零率", "偏度", "峰度"]
    freq_stats = ["主频(Hz)", "频谱能量", "频谱熵"]
    axes = [
        "加速度X轴", "加速度Y轴", "加速度Z轴",
        "陀螺仪X轴", "陀螺仪Y轴", "陀螺仪Z轴",
    ]
    result = {}
    idx = 0
    for ax in axes:
        for stat in time_stats + freq_stats:
            result[f"feat_{idx:03d}"] = f"{ax}{stat}"
            idx += 1
    for stat in time_stats:
        result[f"feat_{idx:03d}"] = f"加速度合力{stat}"
        idx += 1
    for stat in time_stats:
        result[f"feat_{idx:03d}"] = f"角速度合力{stat}"
        idx += 1
    for pair in ["加速度X-Y相关系数", "加速度X-Z相关系数", "加速度Y-Z相关系数"]:
        result[f"feat_{idx:03d}"] = pair
        idx += 1
    return result


FEAT_DESC = _feature_descriptions()

SCENARIO_LABELS = {
    "S1_Normal":      ("S1 Normal",      "训练内"),
    "S2_Active":      ("S2 Active",      "训练内"),
    "S3_Calm":        ("S3 Calm",        "训练内"),
    "S4_Mild_skin":   ("S4 Mild skin",   "训练外"),
    "S5_Severe_skin": ("S5 Severe skin", "训练外"),
}

CLASS_ZH = {
    "movement": "运动 movement",
    "sleep":    "睡眠 sleep",
    "scratch":  "抓挠 scratch",
}


# ── Write evaluation CSVs ─────────────────────────────────────────────────────

def write_scenarios_summary(results: dict) -> None:
    path = EVAL_MODEL / "scenarios_summary.csv"
    rows = []
    for sc, m in results.items():
        label, sc_type = SCENARIO_LABELS[sc]
        sc_data = m["per_class"]["scratch"]
        rows.append({
            "scenario":        label,
            "type":            sc_type,
            "n_samples":       m["n_samples"],
            "accuracy":        round(m["accuracy"], 4),
            "scratch_precision": round(sc_data["precision"], 3),
            "scratch_recall":    round(sc_data["recall"],    3),
            "scratch_f1":        round(sc_data["f1"],        3),
        })
    df = pd.DataFrame(rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 各场景整体准确率汇总\n")
        f.write("# scenario=场景名称  type=训练内/训练外  n_samples=样本数\n")
        f.write("# accuracy=整体准确率  scratch_precision=抓挠精确率  "
                "scratch_recall=抓挠召回率  scratch_f1=抓挠F1\n")
        df.to_csv(f, index=False)
    print(f"  ✓ {path.name}")


def write_classification_report(results: dict) -> None:
    path = EVAL_MODEL / "classification_report.csv"
    rows = []
    for sc, m in results.items():
        label, sc_type = SCENARIO_LABELS[sc]
        for cls, metrics in m["per_class"].items():
            rows.append({
                "scenario":  label,
                "type":      sc_type,
                "class":     CLASS_ZH.get(cls, cls),
                "precision": round(metrics["precision"], 3),
                "recall":    round(metrics["recall"],    3),
                "f1":        round(metrics["f1"],        3),
                "support":   metrics["support"],
            })
    df = pd.DataFrame(rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 各场景各类别详细分类指标\n")
        f.write("# precision=精确率(误报率指标)  recall=召回率(漏报率指标)  "
                "f1=F1分数(综合指标)  support=真实样本数\n")
        df.to_csv(f, index=False)
    print(f"  ✓ {path.name}")


def write_confusion_matrix(results: dict) -> None:
    path = EVAL_MODEL / "confusion_matrix.csv"
    rows = []
    for sc, m in results.items():
        label, sc_type = SCENARIO_LABELS[sc]
        cm      = m["confusion_matrix"]
        cls_names = ["movement", "sleep", "scratch"]
        for i, true_cls in enumerate(cls_names):
            row = {
                "scenario":   label,
                "type":       sc_type,
                "true_class": CLASS_ZH.get(true_cls, true_cls),
            }
            for j, pred_cls in enumerate(cls_names):
                row[f"pred_{pred_cls}"] = cm[i][j]
            rows.append(row)
    df = pd.DataFrame(rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 混淆矩阵（行=真实类别，列=预测类别）\n")
        f.write("# 对角线=预测正确，非对角线=误分类\n")
        df.to_csv(f, index=False)
    print(f"  ✓ {path.name}")


def write_feature_importance(feature_importance: list, top_n: int = 20) -> None:
    path = EVAL_MODEL / "feature_importance.csv"
    rows = []
    for rank, (feat_id, importance) in enumerate(feature_importance[:top_n], 1):
        rows.append({
            "rank":        rank,
            "feature":     feat_id,
            "description": FEAT_DESC.get(feat_id, feat_id),
            "importance":  round(importance, 1),
        })
    df = pd.DataFrame(rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 模型特征重要性排名（LightGBM 分裂增益）\n")
        f.write("# importance 数值越大表示该特征对分类决策的贡献越高\n")
        df.to_csv(f, index=False)
    print(f"  ✓ {path.name}")


# ── Unit tests ────────────────────────────────────────────────────────────────

MODULE_ZH = {
    "test_api_health":        "健康检查接口，验证 MySQL 和 TDengine 连接状态",
    "test_baseline_algorithm":"基线算法：EWMA 更新、权重分配、置信度计算",
    "test_evaluator_scoring": "评分函数：各维度得分计算、阈值判断、告警分级",
    "test_inference_model":   "推理模型：特征提取、滑动窗口分割、行为事件合并",
    "test_tdengine_helpers":  "TDengine 工具函数：时间戳转换、数据拉取",
}


def run_unit_tests() -> dict:
    """Run pytest and return per-module counts."""
    print("\n运行单元测试 …")
    cmd = [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short", "-q"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    module_stats: dict[str, dict] = {}
    for line in output.splitlines():
        # lines look like: tests/unit/test_foo.py::ClassName::test_bar PASSED
        if " PASSED" in line or " FAILED" in line or " ERROR" in line:
            parts = line.split("::")
            if len(parts) >= 2:
                mod = Path(parts[0]).stem  # e.g. "test_api_health"
                if mod not in module_stats:
                    module_stats[mod] = {"passed": 0, "failed": 0}
                if " PASSED" in line:
                    module_stats[mod]["passed"] += 1
                else:
                    module_stats[mod]["failed"] += 1

    total_passed = sum(v["passed"] for v in module_stats.values())
    total_failed = sum(v["failed"] for v in module_stats.values())
    print(f"  结果: {total_passed} 通过 / {total_failed} 失败")
    if result.returncode != 0 and total_failed == 0:
        print("  [警告] pytest 返回非零退出码，请检查输出：")
        print(output[-2000:])

    return {
        "modules":       module_stats,
        "total_passed":  total_passed,
        "total_failed":  total_failed,
        "returncode":    result.returncode,
    }


def write_unit_test_results(stats: dict) -> None:
    path = EVAL_SERVICE / "unit_test_results.csv"
    rows = []
    for mod, counts in stats["modules"].items():
        total  = counts["passed"] + counts["failed"]
        pct    = f"{counts['passed'] / total * 100:.0f}%" if total else "0%"
        rows.append({
            "test_module": mod,
            "description": MODULE_ZH.get(mod, mod),
            "total":       total,
            "passed":      counts["passed"],
            "failed":      counts["failed"],
            "pass_rate":   pct,
        })
    # Summary row
    t = stats["total_passed"] + stats["total_failed"]
    rows.append({
        "test_module": "合计",
        "description": "",
        "total":       t,
        "passed":      stats["total_passed"],
        "failed":      stats["total_failed"],
        "pass_rate":   f"{stats['total_passed'] / t * 100:.0f}%" if t else "0%",
    })
    df = pd.DataFrame(rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 单元测试通过情况\n")
        f.write("# test_module=测试模块  total=用例总数  passed=通过  "
                "failed=失败  pass_rate=通过率\n")
        df.to_csv(f, index=False)
    print(f"  ✓ {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="一键运行算法评估并更新报告数据文件")
    parser.add_argument("--fresh",    action="store_true", help="强制重新生成合成数据")
    parser.add_argument("--no-unit",  action="store_true", help="跳过单元测试")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("算法服务 — 一键评估")
    print("=" * 60)

    # ── Model evaluation ──────────────────────────────────────────────────────
    print("\n[1/3] 模型训练与评估 …")
    # Import from sibling module; sys.path already has project root
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_1_inference",
        str(TESTS_DIR / "test_1_inference.py"),
    )
    t1_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t1_mod)
    eval_results = t1_mod.run_full_evaluation(force_fresh=args.fresh)

    scenario_results  = eval_results["scenarios"]
    feature_importance = eval_results["feature_importance"]

    print("\n[2/3] 写入评估数据文件 …")
    write_scenarios_summary(scenario_results)
    write_classification_report(scenario_results)
    write_confusion_matrix(scenario_results)
    write_feature_importance(feature_importance)

    # ── Unit tests ────────────────────────────────────────────────────────────
    if not args.no_unit:
        print("\n[3/3] 单元测试 …")
        unit_stats = run_unit_tests()
        write_unit_test_results(unit_stats)
    else:
        print("\n[3/3] 单元测试已跳过（--no-unit）")
        unit_stats = None

    # ── Print summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("评估结果汇总")
    print(f"{'='*60}")
    print(f"\n{'场景':<20} {'类型':<8} {'样本数':>8} {'准确率':>8} {'抓挠F1':>8}")
    print("-" * 58)
    for sc, m in scenario_results.items():
        label, sc_type = SCENARIO_LABELS[sc]
        sc_f1 = m["per_class"]["scratch"]["f1"]
        print(f"  {label:<18} {sc_type:<8} {m['n_samples']:>8,} "
              f"  {m['accuracy']:>6.4f}  {sc_f1:>6.3f}")
    print()
    if unit_stats:
        total = unit_stats["total_passed"] + unit_stats["total_failed"]
        print(f"  单元测试: {unit_stats['total_passed']}/{total} 通过")
    print(f"\n  评估数据已写入: {EVAL_DIR}")
    print(f"  总耗时: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
