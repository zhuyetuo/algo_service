# 单元测试说明

本目录包含 algo_service 各功能模块的 pytest 单元测试，共 **88 个测试用例**。

所有测试均不依赖真实数据库或 TDengine，可在任意环境中独立运行。

---

## 环境准备

```bash
# 在项目根目录执行
cd algo_service

# 安装测试依赖（pytest + pytest-asyncio + httpx）
pip install -r requirements-dev.txt

# 安装项目本体依赖（如未安装）
pip install -r requirements.txt
```

---

## 运行方式

```bash
# 运行全部单元测试
python -m pytest tests/unit/

# 显示每个测试名称
python -m pytest tests/unit/ -v

# 只运行某一文件
python -m pytest tests/unit/test_evaluator_scoring.py -v

# 只运行某一测试类
python -m pytest tests/unit/test_evaluator_scoring.py::TestCalcS1 -v

# 只运行某一测试用例
python -m pytest tests/unit/test_inference_model.py::TestExtractFeatures::test_feature_count -v

# 遇到第一个失败立即停止
python -m pytest tests/unit/ -x
```

---

## 测试文件总览

| 文件 | 覆盖模块 | 用例数 | 是否需要外部服务 |
|------|---------|--------|----------------|
| `test_inference_model.py` | `modules/inference/model.py` | 16 | 否 |
| `test_evaluator_scoring.py` | `modules/assessment/evaluator.py` | 36 | 否 |
| `test_baseline_algorithm.py` | `modules/baseline/updater.py`（算法层） | 6 | 否 |
| `test_tdengine_helpers.py` | `db/tdengine.py` | 7 | 否（网络调用已 mock） |
| `test_api_health.py` | `main.py` `/health` 接口 | 3 | 否（DB 和 TDengine 已 mock） |

---

## 各文件详细说明

### `test_inference_model.py` — 行为识别特征提取

覆盖 `modules/inference/model.py` 的纯函数。

| 测试类 | 测试内容 |
|--------|---------|
| `TestTimeFeatures` | 时域特征输出维度（9 维）、常数输入的均值/标准差/极差、对称数组偏度 ≈ 0 |
| `TestFreqFeatures` | 频域特征输出维度（3 维）、零输入返回全零、纯正弦波主频识别 |
| `TestExtractFeatures` | 输出为一维数组、总维度固定为 93、两个不同窗口维度一致 |
| `TestSegment` | 窗口数量正确、数据不足时返回空列表、每个窗口形状正确 |
| `TestWindowsToEvents` | 空标签返回空列表、单标签合并为 1 个事件、多标签正确切分、时间戳单调递增 |

---

### `test_evaluator_scoring.py` — 皮肤健康评分函数

覆盖 `modules/assessment/evaluator.py` 中所有纯评分函数，无数据库依赖。

| 测试类 | 测试内容 |
|--------|---------|
| `TestEvalPhase` | `eval_phase()` 在各 `valid_days` 边界处返回正确阶段（0/1/2/3） |
| `TestGetThreshold` | 预热期返回 `None`；Phase 1/2/3 各阈值值正确 |
| `TestCalcS1` | z≤0 → 0 分；z=2.0 → 12.5 分；z≥4.0 → 上限 25 分 |
| `TestCalcS2` | wpeb 等于均值 → 0 分；高于均值触发得分；std=0 时使用 floor 不崩溃 |
| `TestCalcS3` | 夜间抓挠次数 × 睡眠碎片化查表验证 5 个组合 |
| `TestCalcS4` | 体温为 None → 0 分；佩戴松脱 >240 分钟 → 0 分；四档体温分级 |
| `TestCalcS6` | 高温/低温/低湿/高湿各自修正值；组合相消；clamp 边界 |
| `TestScoreToLevel` | 总分 0/9.9/10/50/99/100 → 健康等级 0/0/1/5/9/10 |
| `TestGetAlertLevel` | 连续天数 0/1/2/3/4/5/10 → 告警等级 0/0/1/2/2/3/3 |

---

### `test_baseline_algorithm.py` — 基线 EWMA 算法

验证 `modules/baseline/updater.py` 的核心算法逻辑，通过本地辅助函数模拟 EWMA 更新过程，不调用数据库。

| 测试类 | 测试内容 |
|--------|---------|
| `TestWeightAssignment` | z<1.0 → weight=0.10；1.0≤z<2.0 → 0.03；z≥2.0 → 0.00（完全冻结） |
| `TestEwmaConvergence` | 90 天正常数据后基线误差 < 1.0；疾病期冻结后基线偏移 < 0.5；valid_days 计数正确；置信度公式 `min(vd/30, 1.0)`；Phase 边界 0/1/2/3 |

---

### `test_tdengine_helpers.py` — TDengine 工具函数

覆盖 `db/tdengine.py`，网络调用部分使用 `unittest.mock` 替换。

| 测试类 | 测试内容 |
|--------|---------|
| `TestTsToMs` | `int` 直接透传；字符串转 `int`；`datetime` 转 UTC 毫秒；含微秒的 `datetime` |
| `TestTdGetDevices` | mock `taosrest.connect` 后返回正确设备列表，且调用 `conn.close()` |
| `TestTdFetch` | 返回包含正确字段的行列表；无数据时返回空列表 |

---

### `test_api_health.py` — `/health` 接口

对 `main.py` 的 `/health` 端点做 HTTP 级别测试，使用 `fastapi.testclient.TestClient`。启动依赖（DB init、调度器）已 mock，每个测试单独 mock DB 和 TDengine 连接。

| 测试 | 场景 | 预期结果 |
|------|------|---------|
| `test_health_all_ok` | DB 和 TDengine 均正常 | `{"status":"ok","postgres":"ok","tdengine":"ok"}` |
| `test_health_postgres_down` | DB 连接抛出异常 | `status="degraded"`，`postgres` 字段含错误信息 |
| `test_health_tdengine_down` | TDengine HTTP 请求抛出异常 | `status="degraded"`，`tdengine` 字段含错误信息 |

---

## 新增测试指南

### 测试纯函数

```python
# tests/unit/test_my_module.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from modules.my_module import my_function

class TestMyFunction:
    def test_normal_case(self):
        assert my_function(1, 2) == 3

    def test_edge_case(self):
        assert my_function(0, 0) == 0
```

### Mock 数据库的异步函数

```python
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_db_function():
    mock_db = AsyncMock()
    mock_result = AsyncMock()
    mock_result.fetchall.return_value = []
    mock_db.execute.return_value = mock_result
    mock_db.commit = AsyncMock()

    await my_async_db_function(mock_db, "device_001")
    mock_db.commit.assert_called_once()
```

### Mock 网络调用

```python
from unittest.mock import MagicMock, patch

def test_network_function():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [("row1",)]
    mock_conn.cursor.return_value = mock_cursor

    with patch("taosrest.connect", return_value=mock_conn):
        result = my_network_function()

    assert result == ["row1"]
```

---

## 目录结构

```
tests/unit/
├── README.md                    本文件
├── __init__.py
├── conftest.py                  sys.path 配置（自动加载）
├── test_inference_model.py      特征提取 / 窗口分割 / 事件合并
├── test_evaluator_scoring.py    皮肤评分纯函数 / 阶段阈值
├── test_baseline_algorithm.py   EWMA 权重 / 收敛性 / 软冻结
├── test_tdengine_helpers.py     时间戳转换 / mock TDengine 调用
└── test_api_health.py           /health 接口（mock DB + TDengine）
```
