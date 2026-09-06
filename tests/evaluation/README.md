# 评估数据目录说明

本目录存放算法服务的自测评估数据，分为两个子目录：

```
tests/evaluation/
├── model/                        # 算法模型评估数据
│   ├── scenarios_summary.csv     # 各场景整体准确率汇总
│   ├── classification_report.csv # 各场景 × 各类别的精确率/召回率/F1
│   ├── confusion_matrix.csv      # 各场景混淆矩阵（真实类别 vs 预测类别）
│   └── feature_importance.csv    # 模型特征重要性排名
│
└── service/                      # 服务功能验证数据
    ├── unit_test_results.csv     # 单元测试通过情况
    ├── health_check.json         # 健康检查接口返回结果
    └── data_write_check.csv      # 各数据库表写入验证结果
```

---

## 如何替换为真实数据

### 模型评估数据（model/）

**当有真实标注的设备数据时：**

1. 将真实 IMU 数据按格式整理为 CSV（列：ts, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, label）
2. 修改 `tests/test_1_inference.py`，将数据加载部分替换为读取真实数据
3. 重新运行评估：
   ```bash
   docker exec algo_service python tests/test_1_inference.py
   ```
4. 将输出结果对照格式填入对应 CSV 文件，覆盖现有占位数据

**字段含义速查：**

| 字段 | 中文含义 |
|------|---------|
| accuracy | 准确率：预测正确的样本占总样本的比例 |
| precision | 精确率：预测为该类别中实际正确的比例（误报率指标）|
| recall | 召回率：该类别真实样本被识别出来的比例（漏报率指标）|
| f1 | F1 分数：精确率和召回率的调和平均，综合性能指标 |
| support | 该类别在测试集中的真实样本数量 |

---

### 服务验证数据（service/）

**每次部署后更新：**

```bash
# 1. 更新健康检查结果
curl http://localhost:8383/health
# 将返回的 JSON 填入 health_check.json 的 result 字段

# 2. 更新单元测试结果
docker exec algo_service python -m pytest tests/unit/ -v
# 将通过/失败数量更新到 unit_test_results.csv

# 3. 更新数据写入验证
# 登录数据库查询各表记录数，更新 data_write_check.csv 的 has_data / sample_count / result
```

---

> 所有 CSV 文件中以 `#` 开头的行为注释，说明各字段含义，不影响数据读取。  
> 文件中标注 `⚠️` 的位置为需要替换的占位数据。
