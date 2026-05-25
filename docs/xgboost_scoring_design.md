# XGBoost 评分系统设计文档

## 基于机器学习的 C-PVAS 瘙痒评估评分层替换方案

**版本**: v1.0  
**日期**: 2026-05-25  
**状态**: 算法工程内部设计草稿  
**受众**: 算法工程团队  
**关联文档**: `docs/cpvas_design.md`（规则系统原始设计）

---

## 目录

1. [方案概述](#1-方案概述)
2. [数据存储设计](#2-数据存储设计)
3. [标签策略](#3-标签策略)
4. [特征工程](#4-特征工程)
5. [模型设计](#5-模型设计)
6. [SHAP 可解释性输出](#6-shap-可解释性输出)
7. [与规则系统并行运行方案](#7-与规则系统并行运行方案)
8. [个体化微调](#8-个体化微调)
9. [实施时间线与里程碑](#9-实施时间线与里程碑)
10. [风险与应对](#10-风险与应对)
11. [与现有代码的集成点](#11-与现有代码的集成点)

---

## 1. 方案概述

### 1.1 问题陈述

当前规则系统将六个维度（S1–S6）通过专家预设权重加总为 0–100 分，再通过固定阈值映射到 L0–L10：

```
Total_Score = S1(25) + S2(25) + S3(20) + S4(10) + S5(15) + S6(±5)
L_level = floor(Total_Score / 10)  # 0-9→L0, 10-19→L1, ... 90+→L9/L10
```

**核心缺陷**：

1. **权重是专家猜测，非数据验证**：S1=25分、S4=10分的权重设定无实证依据。对某些品种（如柴犬），颈温可能是比抓挠频次更强的信号；对另一些犬，体温传感器几乎无效。
2. **线性加权无法捕获维度交互**：「S1高但S3正常」与「S1高且S3差」在临床意义上截然不同，但在加权求和框架下产生相同分数。「高频抓挠+正常睡眠」更可能是习惯性行为，而「高频抓挠+睡眠严重中断」几乎确定是严重瘙痒。
3. **固定阈值不适应个体差异**：同样的总分60分，对于一只总体抓挠频率低的米格鲁，可能意味着 L8；对于一只本身活跃的柴犬，可能只是 L5。

### 1.2 为什么选择 XGBoost 而非深度学习

| 对比维度 | XGBoost | LSTM/Transformer |
|---------|---------|-----------------|
| **数据量需求** | 数百条记录即可训练 | 通常需要数千条以上 |
| **表格数据性能** | 在结构化/表格数据上经实证优势明显 | 需要特殊设计才能匹配 XGBoost |
| **可解释性** | 原生支持 Feature Importance；与 SHAP 完美集成 | 黑盒程度更深，SHAP 计算成本高 |
| **训练速度** | 分钟级，可频繁重训 | 小时级，迭代成本高 |
| **过拟合风险** | 正则化机制成熟（`reg_alpha`, `reg_lambda`） | 数据少时极易过拟合 |
| **部署复杂度** | 单文件 pkl，无 GPU 依赖 | 需要推理框架，内存占用大 |
| **冷启动** | 规则系统标签即可作为初始训练数据 | 必须等待大量真实标注 |

**结论**：在当前数据规模（预计第一年内 <500 只犬，<50,000 条每日记录）和可解释性强约束下，XGBoost 是最适合的选择。数据规模达到 5 万条以上且有充足 vet 标签后，可以评估是否迁移到时序模型。

### 1.3 方案定位：替换评分层，保留特征提取层

XGBoost **不替换**管线的步骤 1–4（佩戴检测、坐标归一化、行为分类、特征提取），只替换步骤 5 的**评分计算数学**。特征提取管线继续输出 z_peb、z_wpeb 等维度值，XGBoost 直接从这些值学习到 L0–L10 的非线性映射。

```
现有管线：
  Step 4 特征提取 → S1/S2/S3/S4 Z-score + S5问卷 + S6环境
                 → [规则系统] 加权求和 → 固定阈值映射 → L0-L10

替换后：
  Step 4 特征提取 → S1/S2/S3/S4 Z-score + S5问卷 + S6环境 + 时序特征
                 → [XGBoost]  非线性映射 + 交互建模   → L0-L10
```

### 1.4 与规则系统的关系

XGBoost **不立即替换**规则系统，而是**并行运行**，通过三个阶段逐步迁移（详见第7节）。规则系统的输出（`rule_score`、`rule_level`）本身将作为弱监督标签来训练 XGBoost 的冷启动版本。

### 1.5 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据采集层                                │
│  IMU(50Hz) + 颈部温度(1/min) + 环境温湿度(1/5min) + 问卷(每日)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  Step 1-4: 特征提取管线（不变）                   │
│  佩戴检测 → 坐标归一化 → 行为分类(LightGBM) → 特征聚合           │
│  输出: z_peb, z_wpeb, sleep_scratch_ratio, sleep_interrupt_z,   │
│        neck_temp_z, env_temp_corr                                │
└──────────────┬─────────────────────────┬───────────────────────┘
               │                         │
┌──────────────▼───────────┐  ┌──────────▼──────────────────────┐
│   [规则系统] Step 5       │  │   [XGBoost 评分层] 新增          │
│   S1-S6 加权求和          │  │   特征向量构造（含时序特征）      │
│   固定阈值映射             │  │   → XGBoost 预测 L0-L10         │
│   → rule_score(0-100)    │  │   → SHAP 贡献值计算              │
│   → rule_level(L0-L10)   │  │   → 置信度评估                   │
└──────────────┬───────────┘  └──────────┬───────────────────────┘
               │                         │
               └───────────┬─────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│              pet_daily_ml_features 表（新增，本文核心）           │
│  同时存储: rule_score/rule_level + xgb_level + 所有特征 + 标签   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      输出路由（Phase 决策）                       │
│  Phase 1-2: 返回 rule_level（XGBoost 静默运行）                  │
│  Phase 3+:  返回 xgb_level（规则系统作备用）                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据存储设计

### 2.1 设计原则

**核心要求**：在任何模型训练开始之前，必须立即开始存储每日特征记录。数据是模型的燃料，晚存储一天就是损失一天的训练数据。表结构设计必须考虑：

1. 覆盖所有可能的模型输入特征（宁多勿少，字段可以为空）
2. 完整记录所有来源的标签（强/弱/自监督）
3. 支持将来的回测和多版本模型评估

### 2.2 每日特征记录表 Schema

```sql
-- 每日特征记录表
-- 此表是模型训练的主要数据源，必须在 XGBoost 训练前长期积累
CREATE TABLE pet_daily_ml_features (
    -- ── 主键与设备标识 ────────────────────────────────────────────
    id              BIGSERIAL PRIMARY KEY,
    dog_id          VARCHAR(64) NOT NULL,        -- 犬只唯一ID（非设备SN，跨换设备稳定）
    device_sn       VARCHAR(64) NOT NULL,        -- 设备序列号（用于关联原始数据）
    stat_date       DATE        NOT NULL,        -- 评估日期（本地日期，非UTC时间戳）
    stat_date_ts    BIGINT      NOT NULL,        -- 对应UTC毫秒时间戳（与现有表对齐）

    -- ── 佩戴状态元信息 ────────────────────────────────────────────
    phase           SMALLINT    NOT NULL DEFAULT 0,  -- 评估阶段 0/1/2/3
    wear_days       INT         NOT NULL DEFAULT 0,  -- 累计有效佩戴天数（即 valid_days）
    wear_minutes    INT         NOT NULL DEFAULT 0,  -- 当日有效佩戴分钟数
    data_quality    SMALLINT    NOT NULL DEFAULT 0,  -- 0=正常 1=佩戴不足 2=数据缺失

    -- ── S1/S2 核心特征（抓挠频次与强度）─────────────────────────
    z_peb           REAL,   -- S1: 当日PEB数量 Z-score（相对个体基线）
    z_wpeb          REAL,   -- S2: 当日W-PEB加权分 Z-score（已含强度权重）
    peb_count       INT,    -- 当日原始PEB次数（绝对值，用于调试和分析）
    wpeb_score      REAL,   -- 当日原始W-PEB分（绝对值）

    -- ── S3 子指标（夜间静息质量）─────────────────────────────────
    sleep_scratch_ratio  REAL,  -- 夜间抓挠帧 / 夜间有效总帧（0.0-1.0）
    sleep_interrupt_z    REAL,  -- 夜间睡眠中断次数 Z-score（相对个体基线）
    sleep_interrupt_cnt  INT,   -- 夜间中断绝对次数（调试用）
    night_peb_count      INT,   -- 夜间（22:00-06:00）PEB次数

    -- ── S4 颈部体温 ───────────────────────────────────────────────
    neck_temp_z     REAL,   -- 颈温日均值（环境温度修正后）Z-score
    neck_temp_mean  REAL,   -- 颈温日均值（绝对值，°C）
    neck_temp_max   REAL,   -- 颈温日最高值（°C）

    -- ── S6 环境指标 ───────────────────────────────────────────────
    env_temp_corr   REAL,   -- 抓挠行为与THI时间序列的皮尔逊相关系数（S6核心）
    thi_z           REAL,   -- 当日THI相对个体历史基线的Z-score
    env_temp_mean   REAL,   -- 当日环境温度日均值（°C）
    env_humidity_mean REAL, -- 当日环境湿度日均值（%RH）

    -- ── 时序特征（需从历史记录计算，按日批量更新）──────────────────
    consecutive_abnormal_days  INT  DEFAULT 0,
        -- 连续异常天数（连续多少天 z_peb > phase阈值），捕捉趋势恶化
    days_since_last_normal     INT,
        -- 距上次正常天（z_peb <= 0.5）的天数，NULL表示从未正常过
    trend_7d                   REAL,
        -- 近7天 rule_score 的线性回归斜率（每天上升/下降多少分）
        -- 正值=恶化趋势，负值=改善趋势
    trend_30d                  REAL,
        -- 近30天 rule_score 的线性回归斜率（长期趋势）
    peb_7d_mean                REAL,   -- 近7天 PEB 数量滚动均值
    peb_7d_std                 REAL,   -- 近7天 PEB 数量滚动标准差（捕捉波动性）
    wpeb_7d_mean               REAL,   -- 近7天 W-PEB 滚动均值

    -- ── 个体信息（相对静态，但需要存储以支持品种/年龄分析）──────────
    breed_encoded   SMALLINT,
        -- 品种编码（0=未知/其他，1=金毛，2=柴犬，3=边牧，...见品种编码表）
        -- 注意：品种对抓挠频次的基线影响显著，XGBoost 需要此信息
    age_months      SMALLINT,
        -- 犬只月龄（评估时）。幼犬(<12月)和老龄犬(>96月)行为模式差异大

    -- ── 规则系统输出（历史记录，同时作为弱监督标签）──────────────
    rule_score      SMALLINT,   -- 规则系统计算的 0-100 总分
    rule_level      SMALLINT,   -- 规则系统映射的 L0-L10 等级（0-10整数）
    s1_score        SMALLINT,   -- 各维度分项（用于分析规则系统与ML的分歧）
    s2_score        SMALLINT,
    s3_score        SMALLINT,
    s4_score        SMALLINT,
    s5_score        SMALLINT,
    s6_score        SMALLINT,   -- 可为负值

    -- ── 标签字段（三级标签体系，优先级从高到低）──────────────────
    questionnaire_level  SMALLINT,
        -- 追问树问卷输出的主人主观评级（L0-L10，nullable）
        -- 这是弱标签，每次主人填写问卷后更新
    vet_confirmed_level  SMALLINT,
        -- 兽医确认的 PVAS 等级（L0-L10，nullable）
        -- 这是强标签，通过 App 端"就诊记录"流程收集
    vet_confirmed_at     BIGINT,   -- 兽医确认时间戳（ms）
    vet_notes            TEXT,     -- 兽医备注（用于数据质量审查）
    label_source         VARCHAR(16),
        -- 该记录最终用于训练的标签来源：
        -- "vet"          → 兽医确认（最高优先级，weight=1.0）
        -- "questionnaire"→ 问卷填写（次级，weight=0.6）
        -- "rule"         → 规则系统输出（自监督冷启动，weight=0.3）
        -- NULL           → 尚无任何标签
    label_weight         REAL,
        -- 该记录在训练时的 sample_weight（由 label_source 和类别稀缺性共同决定）
        -- 在模型训练脚本中动态计算，此处存储最终使用值便于审计

    -- ── XGBoost 预测输出（并行运行阶段记录，用于监控与回测）───────
    xgb_level            SMALLINT,  -- XGBoost 预测等级（0-10）
    xgb_score_raw        REAL,      -- XGBoost 回归原始输出（浮点，四舍五入前）
    xgb_confidence       REAL,      -- 置信度（基于近邻点密度或留一法误差估计）
    xgb_model_version    VARCHAR(32), -- 生成预测的模型版本号（如 "v1.2.0-20260901"）

    -- ── 审计字段 ──────────────────────────────────────────────────
    created_at      BIGINT NOT NULL,   -- 记录创建时间（ms）
    updated_at      BIGINT NOT NULL,   -- 最后更新时间（ms）
    feature_version VARCHAR(16) DEFAULT '1.0',
        -- 特征计算逻辑版本号，用于识别历史特征与当前逻辑是否兼容

    -- ── 唯一约束 ──────────────────────────────────────────────────
    UNIQUE (dog_id, stat_date)
);

-- 常用查询索引
CREATE INDEX idx_pdmf_dog_date     ON pet_daily_ml_features (dog_id, stat_date DESC);
CREATE INDEX idx_pdmf_label_source ON pet_daily_ml_features (label_source) WHERE label_source IS NOT NULL;
CREATE INDEX idx_pdmf_vet_label    ON pet_daily_ml_features (vet_confirmed_level) WHERE vet_confirmed_level IS NOT NULL;
CREATE INDEX idx_pdmf_stat_date    ON pet_daily_ml_features (stat_date DESC);
```

### 2.3 字段必要性说明

| 字段组 | 必要性理由 |
|-------|-----------|
| `z_peb`, `z_wpeb` | 模型核心输入特征，对应 S1/S2 维度，Z-score 形式已消除个体基线差异 |
| `sleep_scratch_ratio`, `sleep_interrupt_z` | S3 拆分为两个子指标存储，而非聚合为 S3 分，保留了信息量；规则系统的分段函数在聚合时会丢失信息 |
| `neck_temp_z` | S4 信号，个体差异大，Z-score 形式是必须的；绝对温度值无意义 |
| `env_temp_corr` | S6 的核心输入是相关系数而非 THI 本身；XGBoost 可以从相关系数中学到比规则更精细的环境修正逻辑 |
| `consecutive_abnormal_days` | 趋势信息，规则系统完全忽略这一维度；连续3天 L5 与偶发 L5 临床意义完全不同 |
| `trend_7d`, `trend_30d` | 短期和长期趋势斜率，捕捉症状的变化速度，对早期预警至关重要 |
| `breed_encoded`, `age_months` | 个体化修正输入；品种决定了正常行为模式的分布，年龄影响皮肤敏感度 |
| `rule_score`, `rule_level` | 冷启动自监督标签的来源，同时作为并行运行阶段的对比基准 |
| `questionnaire_level` | 弱标签输入；问卷依从性是关键瓶颈，此字段允许我们追踪问卷覆盖率 |
| `vet_confirmed_level` | 强标签；整个系统的 ground truth 来源，可以为 NULL 但当有值时覆盖所有其他标签 |
| `label_source`, `label_weight` | 标签溯源和权重审计；训练时必须能追溯每条样本的标签来源，否则无法调试 |
| `xgb_level`, `xgb_confidence` | 并行运行阶段记录预测值，支持离线评估和分歧分析，不影响当前用户体验 |
| `feature_version` | 特征逻辑迭代时，可以标记哪些历史数据与当前特征定义兼容，避免用旧特征训练新模型 |

---

## 3. 标签策略

### 3.1 三级标签体系

标签质量直接决定模型上限。本系统采用三级标签，优先级从高到低：

```
优先级 1（强标签）: 兽医 PVAS 评级
  来源: App 端就诊记录流程 + 医院合作
  权重: weight = 1.0
  特点: 最可信，但数量少，早期可能极度稀缺

优先级 2（弱标签）: 追问树问卷
  来源: 主人每日填写，L0-L10
  权重: weight = 0.6
  特点: 主观偏差较大（主人习惯化导致低估，焦虑型主人导致高估），
        但数量相对充足

优先级 3（自监督）: 规则系统输出
  来源: 现有 evaluator.py 的 rule_level 字段
  权重: weight = 0.3
  特点: 系统性偏差（即本文要解决的问题），但覆盖率 100%，
        冷启动阶段唯一可用的标签
```

### 3.2 标签合并逻辑

```python
def resolve_label(record: dict) -> tuple[int | None, float, str]:
    """
    返回 (最终标签等级, 样本权重, 标签来源)
    优先级：vet > questionnaire > rule
    """
    if record["vet_confirmed_level"] is not None:
        level = record["vet_confirmed_level"]
        source = "vet"
        base_weight = 1.0

    elif record["questionnaire_level"] is not None:
        level = record["questionnaire_level"]
        source = "questionnaire"
        base_weight = 0.6

        # 额外惩罚：问卷与规则系统相差超过3级时，可能是主人主观误差较大
        if record["rule_level"] is not None:
            gap = abs(level - record["rule_level"])
            if gap >= 4:
                base_weight *= 0.7  # 进一步降权，不直接丢弃

    elif record["rule_level"] is not None:
        level = record["rule_level"]
        source = "rule"
        base_weight = 0.3

    else:
        return None, 0.0, "none"  # 无任何标签，不参与训练

    return level, base_weight, source


def compute_sample_weight(level: int, base_weight: float, class_counts: dict) -> float:
    """
    结合标签来源权重和类别稀缺性，计算最终 sample_weight。
    class_counts: {0: 3200, 1: 1800, ..., 10: 120} 各等级样本数
    """
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    # 反频率权重：少数类获得更高权重
    class_weight = (total / (n_classes * class_counts.get(level, 1)))
    # 综合权重，归一化到 [0.1, 5.0] 范围防止极端值
    combined = base_weight * class_weight
    return float(np.clip(combined, 0.1, 5.0))
```

### 3.3 兽医标签收集机制

**方式一：App 内就诊记录流程**

```
用户点击 "记录就诊" → 填写就诊日期、医院名称 →
系统询问: "兽医是否对瘙痒程度给出了评估？"
  → 是 → 展示 L0-L10 描述，请用户转述兽医评价 → 存入 vet_confirmed_level
  → 否 → 仅记录就诊事件，不录入标签
```

**方式二：高分自动触发标签请求**

当连续3天 `rule_level >= L7` 时，App 推送通知："旺旺的皮肤状况持续不佳，如果近期已就诊，请记录兽医评估结果，帮助我们改进评估准确性。"

**方式三：医院合作**

与合作宠物医院对接，就诊时医生在诊疗系统录入 PVAS 评分，通过 API 回传到我们系统（长期目标，需要商务合作）。

### 3.4 预期标签分布

根据皮肤病就诊数据的行业参考，预计标签分布将严重不平衡：

| 等级范围 | 临床含义 | 预计占比 | 说明 |
|---------|---------|---------|-----|
| L0–L2 | 健康/轻微 | ~70% | 大多数犬只日常处于正常状态 |
| L3–L5 | 轻中度 | ~20% | 季节性过敏高发期会增加 |
| L6–L8 | 中重度 | ~8% | 需要就诊的症状 |
| L9–L10 | 严重 | ~2% | 极少见，但误判代价最高 |

### 3.5 类别不平衡处理

**问题**：L0–L2 约占80%，L6–L10 约占5%，直接训练会导致模型偏向多数类。

**策略组合**：

```python
# 策略1: sample_weight（首选，保留所有数据）
# 在 compute_sample_weight() 中已集成反频率权重
# L6-L10 的样本权重将被放大约 15-40 倍

# 策略2: 顺序回归目标（reg:squarederror 对极端值有天然容忍性）
# 比多分类交叉熵对不平衡更鲁棒，不需要额外的 class_weight

# 策略3: 评估阶段分级（严重等级误判的代价不对称）
# 使用自定义评估指标，对漏报严重瘙痒（FN）施加更高惩罚
def asymmetric_mae(y_true, y_pred):
    """对漏报（预测低于真实值）施加更高惩罚"""
    errors = y_pred - y_true
    penalties = np.where(errors < 0, 2.0 * np.abs(errors), np.abs(errors))
    return np.mean(penalties)

# 策略4: 训练集过采样（仅在 L9-L10 样本极度稀缺时启用）
# 当 L9-L10 样本 < 50 条时，对这些样本进行有噪声的复制
# （在特征值上添加小扰动，模拟传感器噪声，不改变标签）
```

---

## 4. 特征工程

### 4.1 完整特征列表

#### 4.1.1 基础特征（来自每日记录，直接使用）

| 特征名 | 类型 | 说明 | 重要性预期 |
|-------|------|-----|----------|
| `z_peb` | float | S1 Z-score：抓挠频次相对个体基线的偏离 | **极高** - 核心信号 |
| `z_wpeb` | float | S2 Z-score：加权抓挠负担（频次×强度×时长）的偏离 | **极高** - 核心信号 |
| `sleep_scratch_ratio` | float [0,1] | 夜间抓挠帧占比，直接反映夜间瘙痒程度 | **高** |
| `sleep_interrupt_z` | float | 夜间睡眠中断次数 Z-score | **高** - 与 sleep_scratch_ratio 有交互 |
| `neck_temp_z` | float | 颈温 Z-score，局部炎症代理指标 | **中** - 噪声较大 |
| `env_temp_corr` | float [-1,1] | 抓挠行为与环境THI的皮尔逊相关系数 | **中** - 区分环境驱动 vs 病理驱动 |

#### 4.1.2 时序特征（需从历史记录计算）

| 特征名 | 类型 | 计算方式 | 说明 | 重要性预期 |
|-------|------|---------|-----|----------|
| `consecutive_abnormal_days` | int | 连续天数计数（z_peb > 阶段阈值） | 区分偶发与持续性瘙痒；连续3天 L5 与偶发 L5 临床意义截然不同 | **高** |
| `trend_7d` | float | 近7天 rule_score 线性回归斜率 | 短期趋势：正值表示恶化，负值表示改善；相同评分但趋势不同，临床处理不同 | **高** |
| `trend_30d` | float | 近30天 rule_score 线性回归斜率 | 长期趋势：捕捉季节性变化和慢性病进展 | **中** |
| `days_since_last_normal` | int | 距上次 z_peb <= 0.5 的天数 | 捕捉慢性化程度；急性发作 vs 慢性持续 | **中** |
| `peb_7d_mean` | float | 近7天 PEB 次数滚动均值 | 短期水平基准，比单日值更稳定 | **中** |
| `peb_7d_std` | float | 近7天 PEB 次数滚动标准差 | 波动性特征：高波动可能意味着间歇性发作 | **中** |
| `wpeb_7d_mean` | float | 近7天 W-PEB 滚动均值 | 短期强度水平基准 | **中** |

**时序特征计算代码**：

```python
def compute_temporal_features(dog_id: str, stat_date: date, db_session) -> dict:
    """
    从历史 pet_daily_ml_features 计算时序特征。
    在每日批量评估（Step 5）中调用，需在写入当日记录前执行。
    """
    # 获取过去30天历史记录
    history = db_session.execute("""
        SELECT stat_date, z_peb, rule_score, peb_count
        FROM pet_daily_ml_features
        WHERE dog_id = :dog_id
          AND stat_date < :today
          AND data_quality = 0
        ORDER BY stat_date DESC
        LIMIT 30
    """, {"dog_id": dog_id, "today": stat_date}).fetchall()

    if not history:
        return {
            "consecutive_abnormal_days": 0,
            "days_since_last_normal": None,
            "trend_7d": 0.0, "trend_30d": 0.0,
            "peb_7d_mean": None, "peb_7d_std": None,
            "wpeb_7d_mean": None,
        }

    # consecutive_abnormal_days
    consec = 0
    for row in history:  # 已按日期降序
        if row.z_peb is not None and row.z_peb > 1.5:  # 使用统一阈值，不依赖动态阶段
            consec += 1
        else:
            break

    # days_since_last_normal
    days_since_normal = None
    for i, row in enumerate(history):
        if row.z_peb is not None and row.z_peb <= 0.5:
            days_since_normal = i + 1  # +1 因为 history[0] 是昨天
            break

    # trend_7d: 线性回归斜率
    recent_7 = history[:7]
    trend_7d = 0.0
    if len(recent_7) >= 3:
        x = np.arange(len(recent_7))  # 时间索引
        y = np.array([r.rule_score for r in recent_7 if r.rule_score is not None])
        if len(y) >= 3:
            slope, _ = np.polyfit(x[:len(y)], y, 1)
            trend_7d = float(slope)  # 每天的分数变化量

    # trend_30d
    trend_30d = 0.0
    if len(history) >= 7:
        x = np.arange(len(history))
        y = np.array([r.rule_score for r in history if r.rule_score is not None])
        if len(y) >= 7:
            slope, _ = np.polyfit(x[:len(y)], y, 1)
            trend_30d = float(slope)

    # 滚动统计
    peb_vals_7d = [r.peb_count for r in recent_7 if r.peb_count is not None]
    peb_7d_mean = float(np.mean(peb_vals_7d)) if peb_vals_7d else None
    peb_7d_std  = float(np.std(peb_vals_7d))  if len(peb_vals_7d) >= 3 else None

    return {
        "consecutive_abnormal_days": consec,
        "days_since_last_normal": days_since_normal,
        "trend_7d": trend_7d,
        "trend_30d": trend_30d,
        "peb_7d_mean": peb_7d_mean,
        "peb_7d_std": peb_7d_std,
        "wpeb_7d_mean": None,  # 同理从 wpeb_score 计算，略
    }
```

#### 4.1.3 个体信息特征

| 特征名 | 类型 | 说明 | 重要性预期 |
|-------|------|-----|----------|
| `age_months` | int | 月龄：幼犬皮肤更敏感，老龄犬基础抓挠率更高 | **中** |
| `breed_encoded` | int | 品种编码：金毛/拉布拉多有过敏倾向，柴犬皮肤病高发 | **中** |
| `wear_days` | int | 累计佩戴天数，即基线质量指标 | **低-中** - 影响特征可信度 |
| `phase` | int [0,3] | 评估阶段，与 wear_days 高度相关，辅助特征 | **低** |

**品种编码表**（初始版本，可扩展）：

```python
BREED_ENCODING = {
    0:  "unknown",
    1:  "golden_retriever",    # 金毛寻回犬（过敏高发）
    2:  "labrador",            # 拉布拉多（过敏高发）
    3:  "shiba_inu",           # 柴犬（皮肤病高发）
    4:  "border_collie",       # 边牧
    5:  "french_bulldog",      # 法斗（皮肤褶皱处易感染）
    6:  "poodle",              # 泰迪/贵宾
    7:  "corgi",               # 柯基
    8:  "husky",               # 哈士奇
    9:  "samoyed",             # 萨摩耶
    10: "other_large",         # 其他大型犬（>25kg）
    11: "other_medium",        # 其他中型犬（10-25kg）
    12: "other_small",         # 其他小型犬（<10kg）
}
```

### 4.2 维度交互期望

以下是我们**期望** XGBoost 通过分裂树节点自动学习的交互效应。记录在此是为了在 SHAP 分析阶段验证模型是否确实学到了这些模式：

| 交互组合 | 期望行为 | 临床依据 |
|---------|---------|---------|
| `z_peb` 高 + `sleep_interrupt_z` 高 | 预测等级远高于两者单独贡献之和 | 夜间睡眠中断是严重瘙痒的强确认信号，而非独立噪声 |
| `z_peb` 高 + `sleep_interrupt_z` 正常 | 预测等级适度，不应线性外推 | 白天高频抓挠但夜间正常，可能是习惯性行为或轻度过敏 |
| `neck_temp_z` 高 + `env_temp_corr` 高 | 分别降低颈温的权重 | 环境热应激可以解释颈温偏高，不应归因于皮肤炎症 |
| `neck_temp_z` 高 + `env_temp_corr` 低 | 放大颈温信号权重 | 在环境正常时颈温偏高，更可能是真实的局部炎症信号 |
| `consecutive_abnormal_days` >= 5 + 任何 z_peb | 预测等级上调 | 持续异常的累积效应在规则系统中完全被忽略 |
| `trend_7d` 快速上升 + 当前 L4 | 预测接近 L5，提前预警 | 快速恶化的 L4 比稳定的 L4 临床紧迫性更高 |
| `questionnaire_level`（S5）高 + 传感器信号中等 | 上调预测，信任主人观察 | 主人能观察到传感器盲区（腹部抓挠、皮肤外观等） |

---

## 5. 模型设计

### 5.1 问题建模方式

将 L0–L10 的预测建模为**有序回归（Ordinal Regression）**，而非11类分类问题：

- **为什么不用多分类**：L0 和 L1 的差异与 L0 和 L10 的差异本质不同，交叉熵损失无法表达这种有序性。L0 和 L10 的混淆代价远大于 L5 和 L6 的混淆。
- **实现方式**：使用 `reg:squarederror` 目标，将输出四舍五入到最近整数，并 clamp 到 [0, 10]。这在实践中对有序标签效果优于直接多分类。
- **替代方案**：`reg:absoluteerror`（MAE 损失）对离群标签更鲁棒，可在调参阶段对比。

### 5.2 完整训练代码

```python
"""
xgboost_trainer.py

XGBoost C-PVAS 评分模型训练脚本。
运行方式: python -m scripts.xgboost_trainer --label-source all --output weights/scoring_xgb_v1.pkl

依赖: xgboost>=2.0, optuna>=3.0, shap>=0.44, scikit-learn>=1.4
"""

import pickle
import logging
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 特征列定义（顺序固定，推理时必须与此一致）
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    # S1/S2 核心信号
    "z_peb",
    "z_wpeb",
    # S3 子指标
    "sleep_scratch_ratio",
    "sleep_interrupt_z",
    # S4 颈温
    "neck_temp_z",
    # S6 环境
    "env_temp_corr",
    # 时序特征
    "consecutive_abnormal_days",
    "trend_7d",
    "trend_30d",
    "days_since_last_normal",
    "peb_7d_mean",
    "peb_7d_std",
    "wpeb_7d_mean",
    # 个体信息
    "age_months",
    "breed_encoded",
    "wear_days",
    "phase",
]

# 有序回归超参数基础配置（调参前的合理初值）
BASE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,             # 浅树减少过拟合，特征数量有限时4足够
    "learning_rate": 0.05,      # 较小学习率配合 n_estimators=300
    "subsample": 0.8,           # 行采样，防止过拟合
    "colsample_bytree": 0.8,    # 列采样，增加多样性
    "reg_alpha": 0.1,           # L1 正则，促进特征稀疏
    "reg_lambda": 1.0,          # L2 正则
    "objective": "reg:squarederror",  # 有序回归核心
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",      # 更快的直方图算法，对表格数据足够
}


def load_training_data(
    db_url: str,
    min_date: date | None = None,
    label_sources: list[str] = ("vet", "questionnaire", "rule"),
    min_wear_days: int = 3,  # 过滤掉基线太少的数据，特征噪声太大
) -> pd.DataFrame:
    """
    从数据库加载训练数据。
    仅加载有 label_source 的记录（label_source IS NOT NULL）。
    """
    import sqlalchemy as sa

    query = """
        SELECT
            dog_id,
            {features},
            CASE
                WHEN vet_confirmed_level IS NOT NULL THEN vet_confirmed_level
                WHEN questionnaire_level IS NOT NULL THEN questionnaire_level
                ELSE rule_level
            END AS target_level,
            label_source,
            label_weight
        FROM pet_daily_ml_features
        WHERE label_source = ANY(:sources)
          AND data_quality = 0
          AND wear_days >= :min_wear_days
          {date_filter}
        ORDER BY dog_id, stat_date
    """.format(
        features=", ".join(FEATURES),
        date_filter=f"AND stat_date >= '{min_date}'" if min_date else "",
    )

    engine = sa.create_engine(db_url)
    df = pd.read_sql(query, engine, params={
        "sources": list(label_sources),
        "min_wear_days": min_wear_days,
    })

    # 缺失值处理
    # days_since_last_normal: NULL 表示从未正常，填充一个大值（如60天）
    df["days_since_last_normal"] = df["days_since_last_normal"].fillna(60)
    # 其他时序特征：用0填充（表示无历史信息）
    for col in ["trend_7d", "trend_30d", "peb_7d_mean", "peb_7d_std", "wpeb_7d_mean"]:
        df[col] = df[col].fillna(0.0)
    # 颈温：有一定缺失率，用0填充（Z-score=0意味着无异常信息）
    df["neck_temp_z"] = df["neck_temp_z"].fillna(0.0)

    logger.info(
        "加载训练数据: %d 条记录, %d 只犬, 标签分布:\n%s",
        len(df),
        df["dog_id"].nunique(),
        df["target_level"].value_counts().sort_index().to_string(),
    )
    return df


def make_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    按 dog_id 分组的 K 折交叉验证。

    关键原则：同一只犬的所有历史记录必须在同一折中。
    原因：
    1. 防止数据泄露 - 同一只犬的相邻日期数据之间存在强自相关
       （今天的特征依赖昨天的历史，如果拆开会导致验证集泄露训练集信息）
    2. 评估真实泛化能力 - 我们关心的是模型能否推广到新的犬只，
       而不是能否记住已见过犬只的特定模式
    3. 反映真实部署场景 - 实际使用时，新用户的犬只从未出现在训练集中
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["dog_id"].values
    splits = list(gkf.split(df, groups=groups))
    return splits


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    模型评估指标：
    1. MAE on level: 平均绝对误差（主要指标）
    2. within_1_acc: 预测值与真实值差距 ≤1 的比例
    3. within_2_acc: 差距 ≤2 的比例
    4. severe_recall: L6-L10 样本中，预测 >= L5 的比例（严重漏报率）
    """
    raw_pred = model.predict(X)
    y_pred = np.clip(np.round(raw_pred), 0, 10).astype(int)

    mae = mean_absolute_error(y, y_pred)
    within_1 = np.mean(np.abs(y_pred - y) <= 1)
    within_2 = np.mean(np.abs(y_pred - y) <= 2)

    severe_mask = y >= 6
    severe_recall = (
        np.mean(y_pred[severe_mask] >= 5)
        if severe_mask.sum() > 0
        else float("nan")
    )

    return {
        "mae": round(mae, 3),
        "within_1_acc": round(within_1, 4),
        "within_2_acc": round(within_2, 4),
        "severe_recall": round(severe_recall, 4),
        "n_samples": len(y),
    }


def train_with_cv(
    df: pd.DataFrame,
    params: dict,
    n_splits: int = 5,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    带 GroupKFold 交叉验证的训练函数。
    返回 (在全量数据上训练的最终模型, 各折评估指标)。
    """
    X = df[FEATURES].values.astype(np.float32)
    y = df["target_level"].values.astype(np.float32)
    w = df["label_weight"].values.astype(np.float32)

    splits = make_group_kfold_splits(df, n_splits)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = w[train_idx]

        fold_model = xgb.XGBRegressor(**params)
        fold_model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        metrics = evaluate_model(fold_model, X_val, y_val.astype(int))
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        logger.info(
            "Fold %d: MAE=%.3f, within1=%.3f, severe_recall=%.3f",
            fold_idx, metrics["mae"], metrics["within_1_acc"], metrics["severe_recall"],
        )

    # 在全量数据上训练最终模型
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X, y, sample_weight=w, verbose=False)

    avg_metrics = {
        "cv_mae_mean": np.mean([m["mae"] for m in fold_metrics]),
        "cv_mae_std":  np.std([m["mae"]  for m in fold_metrics]),
        "cv_within1":  np.mean([m["within_1_acc"]  for m in fold_metrics]),
        "cv_within2":  np.mean([m["within_2_acc"]  for m in fold_metrics]),
        "cv_severe_recall": np.mean([
            m["severe_recall"] for m in fold_metrics
            if not np.isnan(m["severe_recall"])
        ]),
        "fold_details": fold_metrics,
    }

    return final_model, avg_metrics


def optuna_objective(trial, df: pd.DataFrame) -> float:
    """
    Optuna 超参数搜索目标函数。
    优化目标：最小化 CV MAE。
    """
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "max_depth":        trial.suggest_int("max_depth", 3, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective":        "reg:squarederror",
        "random_state":     42,
        "n_jobs":           -1,
        "tree_method":      "hist",
    }
    _, metrics = train_with_cv(df, params, n_splits=5)
    return metrics["cv_mae_mean"]


def tune_hyperparameters(df: pd.DataFrame, n_trials: int = 50) -> dict:
    """运行 Optuna 超参数调优，返回最佳参数。"""
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    logger.info("最佳 CV MAE: %.3f", study.best_value)
    logger.info("最佳参数: %s", study.best_params)
    return {**BASE_PARAMS, **study.best_params}


def train_and_save(
    db_url: str,
    output_path: str = "weights/scoring_xgb_v1.pkl",
    tune: bool = False,
    label_sources: tuple = ("vet", "questionnaire", "rule"),
):
    """主训练入口。"""
    df = load_training_data(db_url, label_sources=label_sources)

    if len(df) < 100:
        raise ValueError(
            f"训练数据不足（{len(df)} 条）。至少需要100条记录才能训练有意义的模型。"
            "继续积累数据，或降低 min_wear_days 要求。"
        )

    if tune:
        best_params = tune_hyperparameters(df, n_trials=50)
    else:
        best_params = BASE_PARAMS

    model, cv_metrics = train_with_cv(df, best_params)

    # 保存模型和元信息
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "features": FEATURES,
        "cv_metrics": cv_metrics,
        "params": best_params,
        "training_samples": len(df),
        "training_dogs": df["dog_id"].nunique(),
        "label_distribution": df["target_level"].value_counts().to_dict(),
        "model_version": f"v1.0.0-{date.today().strftime('%Y%m%d')}",
    }
    with open(output, "wb") as f:
        pickle.dump(artifact, f)

    logger.info("模型已保存至 %s", output)
    logger.info(
        "CV 评估: MAE=%.3f±%.3f, within1=%.1f%%, severe_recall=%.1f%%",
        cv_metrics["cv_mae_mean"],
        cv_metrics["cv_mae_std"],
        cv_metrics["cv_within1"] * 100,
        cv_metrics["cv_severe_recall"] * 100,
    )
    return artifact
```

### 5.3 模型验收标准（Go/No-Go 指标）

| 指标 | 冷启动版本（rule标签）| 弱标签版本（+问卷）| 强标签版本（+兽医）|
|-----|--------------------|--------------------|------------------|
| CV MAE | ≤ 1.5 级 | ≤ 1.2 级 | ≤ 0.8 级 |
| within-1 准确率 | ≥ 70% | ≥ 78% | ≥ 85% |
| L6-L10 召回率 | ≥ 60% | ≥ 70% | ≥ 80% |
| 每只犬样本数 | ≥ 3 | ≥ 5 | ≥ 5 |

---

## 6. SHAP 可解释性输出

### 6.1 为什么 SHAP 是强制要求

在医疗辅助场景中，"黑盒"输出是不可接受的。用户和兽医需要理解为什么模型给出 L6 而不是 L3。SHAP（SHapley Additive exPlanations）提供了理论上有保证的特征贡献解释，且与 XGBoost 的树结构高度兼容，计算速度快。

**要求**：每次 XGBoost 预测必须同时计算 SHAP 值，转换为用户可读的中文解释，存入数据库，并在 App 中展示。

### 6.2 SHAP 计算与转换管线

```python
"""
shap_explainer.py

SHAP 贡献值计算与用户可读解释生成。
"""

import shap
import numpy as np
import pickle
from dataclasses import dataclass, field

# 特征到中文名称的映射
FEATURE_CN_NAMES = {
    "z_peb":                    "抓挠频次异常程度",
    "z_wpeb":                   "抓挠强度综合负担",
    "sleep_scratch_ratio":      "夜间睡眠中抓挠比例",
    "sleep_interrupt_z":        "睡眠中断异常程度",
    "neck_temp_z":              "颈部体温偏高程度",
    "env_temp_corr":            "行为与环境热应激相关性",
    "consecutive_abnormal_days":"连续异常天数（趋势）",
    "trend_7d":                 "近7天症状趋势",
    "trend_30d":                "近30天长期趋势",
    "days_since_last_normal":   "距上次正常状态天数",
    "peb_7d_mean":              "近7天平均抓挠频次",
    "peb_7d_std":               "近7天抓挠波动性",
    "wpeb_7d_mean":             "近7天平均抓挠强度",
    "age_months":               "犬只年龄",
    "breed_encoded":            "品种特性",
    "wear_days":                "项圈佩戴历史长度",
    "phase":                    "基线成熟度",
}


@dataclass
class FeatureContribution:
    feature_name: str
    cn_name: str
    shap_value: float      # 对预测等级的贡献（正=升高预测，负=降低预测）
    feature_value: float   # 该特征的实际数值
    direction: str         # "up" / "down" / "neutral"
    description: str       # 用户可读描述


@dataclass
class ExplanationResult:
    dog_name: str
    predicted_level: int
    raw_score: float
    base_value: float       # SHAP 基准值（训练集平均预测）
    confidence: float
    contributions: list[FeatureContribution] = field(default_factory=list)
    data_quality_note: str = ""

    def to_display_text(self) -> str:
        """生成 App 展示用的中文文本。"""
        lines = [
            f"{self.dog_name}今日皮肤健康评估：L{self.predicted_level}",
            "",
            "主要原因：",
        ]
        # 只展示 SHAP 绝对值 >= 0.1 的特征，最多5个
        significant = [c for c in self.contributions if abs(c.shap_value) >= 0.1][:5]
        for c in significant:
            lines.append(f"  {c.description}")

        lines.append("")
        lines.append(
            f"模型置信度：{int(self.confidence * 100)}%  |  "
            f"{self.data_quality_note}"
        )
        return "\n".join(lines)


class ScoringExplainer:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self.features = artifact["features"]
        self.model_version = artifact.get("model_version", "unknown")

        # TreeExplainer 是 XGBoost 的精确 SHAP 计算器，无需采样近似
        self.explainer = shap.TreeExplainer(self.model)

    def explain(
        self,
        X_single_day: np.ndarray,   # shape: (1, n_features)
        dog_name: str,
        wear_days: int,
        questionnaire_filled: bool,
    ) -> ExplanationResult:
        """
        计算单条记录的 SHAP 解释。

        X_single_day: 特征向量，顺序必须与 self.features 一致
        """
        assert X_single_day.shape == (1, len(self.features)), \
            f"特征维度不匹配: 期望 (1, {len(self.features)})，得到 {X_single_day.shape}"

        # 计算原始预测
        raw_pred = float(self.model.predict(X_single_day)[0])
        predicted_level = int(np.clip(round(raw_pred), 0, 10))

        # 计算 SHAP 值
        shap_values = self.explainer.shap_values(X_single_day)[0]  # shape: (n_features,)
        base_value = float(self.explainer.expected_value)

        # 置信度估算（简化版：基于预测值距最近整数边界的距离）
        # 例：预测 5.8 的置信度高于预测 5.1（距离 5.5 边界更远）
        dist_to_boundary = abs(raw_pred - round(raw_pred))
        confidence = 0.5 + min(dist_to_boundary * 1.5, 0.45)  # 范围 [0.5, 0.95]

        # 生成每个特征的贡献描述
        contributions = self._make_contributions(
            shap_values, X_single_day[0], predicted_level
        )

        # 数据质量说明
        quality_notes = []
        if wear_days >= 60:
            quality_notes.append(f"数据质量：高（佩戴{wear_days}天")
        elif wear_days >= 30:
            quality_notes.append(f"数据质量：中（佩戴{wear_days}天")
        else:
            quality_notes.append(f"数据质量：低（佩戴{wear_days}天，基线尚不充分")
        quality_notes.append("，问卷已填）" if questionnaire_filled else "，问卷未填）")
        data_quality_note = "".join(quality_notes)

        # 按 SHAP 绝对值降序排列
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

        return ExplanationResult(
            dog_name=dog_name,
            predicted_level=predicted_level,
            raw_score=round(raw_pred, 2),
            base_value=round(base_value, 2),
            confidence=round(confidence, 2),
            contributions=contributions,
            data_quality_note=data_quality_note,
        )

    def _make_contributions(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        predicted_level: int,
    ) -> list[FeatureContribution]:
        contributions = []
        for i, feat_name in enumerate(self.features):
            sv = float(shap_values[i])
            fv = float(feature_values[i])
            cn_name = FEATURE_CN_NAMES.get(feat_name, feat_name)

            if abs(sv) < 0.05:
                direction = "neutral"
                symbol = "→"
            elif sv > 0:
                direction = "up"
                symbol = "↑"
            else:
                direction = "down"
                symbol = "↓"

            description = self._generate_description(
                feat_name, cn_name, sv, fv, symbol
            )

            contributions.append(FeatureContribution(
                feature_name=feat_name,
                cn_name=cn_name,
                shap_value=round(sv, 2),
                feature_value=round(fv, 3),
                direction=direction,
                description=description,
            ))
        return contributions

    def _generate_description(
        self, feat_name: str, cn_name: str, sv: float, fv: float, symbol: str
    ) -> str:
        """根据特征名和数值生成人类可读的中文描述。"""
        sv_str = f"{sv:+.1f}级"

        if feat_name == "z_peb":
            times = f"{fv:.1f}倍" if fv > 0 else "低于平时"
            return f"  {symbol} 抓挠频次异常（Z={fv:.2f}，比平时高{times}）    贡献 {sv_str}"

        elif feat_name == "sleep_interrupt_z":
            cnt_info = f"Z={fv:.2f}"
            return f"  {symbol} 睡眠质量下降（中断{cnt_info}）        贡献 {sv_str}"

        elif feat_name == "neck_temp_z":
            deg = f"+{fv*0.5:.1f}°C" if fv > 0 else f"{fv*0.5:.1f}°C"
            return f"  {symbol} 颈温持续偏高（静息状态{deg}）              贡献 {sv_str}"

        elif feat_name == "consecutive_abnormal_days":
            return f"  {symbol} 连续异常已{int(fv)}天（趋势加重）                    贡献 {sv_str}"

        elif feat_name == "env_temp_corr":
            if abs(fv) < 0.3:
                return f"  → 环境温湿度正常（无环境驱动因素）              贡献  0.0级"
            elif fv > 0.5:
                return f"  ↓ 环境热应激较高（部分抓挠可能由环境引起）       贡献 {sv_str}"
            else:
                return f"  {symbol} 环境因素：{cn_name}（{fv:.2f}）                   贡献 {sv_str}"

        else:
            return f"  {symbol} {cn_name}（值={fv:.2f}）                        贡献 {sv_str}"
```

### 6.3 示例输出：旺旺（金毛，45天，预测 L6）

**输入特征值**：

```python
features = {
    "z_peb": 2.55,               # 抓挠频次比基线高2.55倍标准差
    "z_wpeb": 2.41,              # 加权抓挠负担同样显著偏高
    "sleep_scratch_ratio": 0.082, # 夜间8.2%时间在抓挠（远超正常1%）
    "sleep_interrupt_z": 2.67,   # 夜间中断次数比基线高2.67倍标准差
    "neck_temp_z": 2.40,         # 颈温比基线高2.4倍标准差（约+1.2°C）
    "env_temp_corr": 0.12,       # 与环境相关性低，排除环境因素
    "consecutive_abnormal_days": 3,
    "trend_7d": 4.5,             # 每天上升4.5分，快速恶化趋势
    ...
}
```

**SHAP 输出**（示例）：

```
旺旺今日皮肤健康评估：L6

主要原因：
  ↑ 抓挠频次异常（Z=2.55，比平时高2.5倍）    贡献 +2.1级
  ↑ 睡眠质量下降（中断Z=2.67，比平时多8倍）   贡献 +1.3级
  ↑ 颈温持续偏高（静息状态+1.2°C）              贡献 +0.8级
  ↑ 连续异常已3天（趋势加重）                    贡献 +0.5级
  → 环境温湿度正常（无环境驱动因素）              贡献  0.0级

模型置信度：87%  |  数据质量：高（佩戴45天，问卷已填）
```

**解释逻辑验证**：

- 基准值（训练集平均预测）= L1.5（大多数犬处于轻微状态）
- 各贡献之和：1.5 + 2.1 + 1.3 + 0.8 + 0.5 + 0.0 + 其他特征 ≈ 6.0 ✓
- 环境相关性低（0.12）→ 贡献接近0 → 正确排除环境驱动因素

---

## 7. 与规则系统并行运行方案

### 7.1 三阶段迁移策略

#### Phase 1（月份 1–3）：静默运行阶段

**描述**：XGBoost 模型在后台计算预测值，结果仅写入 `pet_daily_ml_features.xgb_level`，不对用户展示，不影响任何现有功能。

**目标**：
- 验证特征计算管线端到端是否正确
- 观察 XGBoost 输出与规则系统的分歧分布
- 积累数据，为下一阶段准备充足的分析基础

**触发条件进入 Phase 2**：
- XGBoost 在内部评估集（有 questionnaire_level 的记录）上 MAE ≤ 1.5 级
- 运行超过60天，至少覆盖100只犬

#### Phase 2（月份 4–6）：内部双轨展示阶段

**描述**：在内部管理后台同时展示两个系统的结果，供算法团队和医疗团队对比分析。

**分歧监控**：

```python
def check_divergence(rule_level: int, xgb_level: int, dog_id: str) -> dict:
    """
    分析两系统的分歧情况。
    当分歧 >= 2 级时，该记录进入"优先标注队列"，优先获取兽医确认标签。
    """
    divergence = abs(xgb_level - rule_level)

    result = {
        "rule_level": rule_level,
        "xgb_level": xgb_level,
        "divergence": divergence,
        "dog_id": dog_id,
        "priority_label_needed": divergence >= 2,
    }

    if divergence >= 2:
        logger.warning(
            "DIVERGENCE_ALERT dog=%s rule=L%d xgb=L%d diff=%d — 加入优先标注队列",
            dog_id, rule_level, xgb_level, divergence,
        )
        # TODO: 触发 App 内"就诊记录"提示，或通知运营人员主动联系用户

    return result
```

**分歧分析要点**：
- 追踪 `|xgb_level - rule_level| >= 2` 的比例（预计 10–20%）
- 分析分歧方向：XGBoost 倾向于高估还是低估（相对规则系统）
- 对分歧最大的 Top 50 个案例进行人工复查
- 分析哪些特征值组合下分歧最大（提示规则系统最不准确的场景）

**触发条件进入 Phase 3**：
- 对至少 200 条有 questionnaire_level 的记录，XGBoost MAE ≤ 规则系统 MAE
- 有至少 50 条 vet_confirmed_level 记录，XGBoost MAE ≤ 1.0 级
- 医疗团队审查分歧案例后确认 XGBoost 判断更合理的比例 ≥ 60%

#### Phase 3（月份 6+）：XGBoost 主导阶段

**描述**：XGBoost 成为主评分系统，用户看到的 L0–L10 来自 XGBoost。规则系统作为后备和监控工具继续运行。

**后备启用条件（回退到规则系统）**：

```python
def should_use_rule_fallback(record: dict, xgb_confidence: float) -> tuple[bool, str]:
    """
    判断是否应该使用规则系统的后备输出。
    返回 (是否使用后备, 原因描述)
    """
    # 条件1: 犬只数据积累不足，XGBoost 对个体模式不熟悉
    if record["wear_days"] < 14:
        return True, "佩戴时间不足14天，XGBoost个体基线尚不稳定"

    # 条件2: XGBoost 置信度低（预测值接近两个等级边界中点）
    if xgb_confidence < 0.60:
        return True, f"模型置信度低（{int(xgb_confidence*100)}%），退回规则系统"

    # 条件3: 核心特征缺失过多，无法信任预测
    missing_critical = sum([
        record.get("neck_temp_z") is None,
        record.get("sleep_interrupt_z") is None,
        record.get("questionnaire_level") is None,  # S5问卷缺失
    ])
    if missing_critical >= 2:
        return True, "关键特征缺失，数据不完整"

    # 条件4: 特征值超出训练分布范围（检测极端外推）
    if abs(record.get("z_peb", 0)) > 6.0:
        return True, "特征值超出训练分布范围，预测不可靠"

    return False, ""
```

### 7.2 监控体系

```python
# 每日监控指标（存入监控表，定期绘制趋势图）
MONITORING_METRICS = {
    # 系统健康
    "xgb_prediction_rate":      "每日使用XGBoost的记录占比",
    "fallback_rate":             "退回规则系统的比例",
    "high_confidence_rate":      "置信度 >= 0.80 的比例",

    # 准确性（有标签时）
    "xgb_mae_vs_questionnaire":  "XGBoost vs 问卷标签的 MAE",
    "rule_mae_vs_questionnaire": "规则系统 vs 问卷标签的 MAE（对比基准）",
    "severe_false_negative_rate": "真实L6+被预测为L3-以下的比例",
    "severe_false_positive_rate": "真实L0-3被预测为L6+的比例",

    # 分歧监控
    "divergence_rate_ge2":        "两系统分歧 >= 2 级的比例",
    "divergence_rate_ge3":        "两系统分歧 >= 3 级的比例（需要立即人工复查）",
}
```

**告警规则**：

| 指标 | 告警阈值 | 响应动作 |
|-----|---------|---------|
| 每日预测失败率 | > 5% | 立即检查特征计算管线 |
| fallback_rate | > 30% | 检查新用户比例是否异常高，或特征缺失率上升 |
| divergence_rate_ge3 | > 10% | 人工审查最近7天的分歧案例 |
| severe_false_negative_rate | > 15% | 立即停止 Phase 3 切换，回滚到规则系统 |

---

## 8. 个体化微调

### 8.1 两层架构设计

```
┌───────────────────────────────────────────────┐
│          Layer 1: 全局 XGBoost 模型             │
│  训练数据: 所有犬只历史记录                      │
│  学习内容: 特征 → L0-L10 的通用非线性映射        │
│  更新频率: 每季度或积累1000条新标签后重训         │
└──────────────────────┬────────────────────────┘
                       │ 全局模型输出（浮点预测值）
┌──────────────────────▼────────────────────────┐
│         Layer 2: 个体校准层（每只犬）            │
│  类型: 单参数线性偏置校正                        │
│        y_calibrated = y_global + bias_i        │
│  参数: bias_i（该犬的预测偏差修正，初始=0）       │
│  训练数据: 仅该犬自身的有标签记录                 │
│  激活条件: wear_days >= 60 AND 有标签记录 >= 5   │
└──────────────────────┬────────────────────────┘
                       │ 最终输出
                  L0-L10 预测值
```

### 8.2 个体校准层实现

```python
class IndividualCalibrator:
    """
    单犬个体校准层。
    使用简单的偏置项校正全局模型对特定犬只的系统性误差。
    例如：全局模型总是低估某只柴犬 1.2 级 → bias = +1.2
    """

    def __init__(self):
        self._bias_store: dict[str, float] = {}  # dog_id → bias
        self._sample_counts: dict[str, int] = {}  # dog_id → 标签样本数

    def update(self, dog_id: str, global_predictions: list[float], true_labels: list[int]):
        """
        更新指定犬只的校准偏置。

        使用中位数残差作为偏置（比均值更鲁棒，抗异常值）。
        """
        if len(true_labels) < 5:
            return  # 样本不足，不更新

        residuals = [float(true) - float(pred)
                     for true, pred in zip(true_labels, global_predictions)]
        new_bias = float(np.median(residuals))

        # 平滑更新（防止单次大更新）
        old_bias = self._bias_store.get(dog_id, 0.0)
        alpha = min(0.3, len(true_labels) / 20)  # 样本越多，更新越快
        self._bias_store[dog_id] = old_bias * (1 - alpha) + new_bias * alpha
        self._sample_counts[dog_id] = len(true_labels)

    def predict(
        self,
        dog_id: str,
        global_prediction: float,
        wear_days: int,
        labeled_sample_count: int,
    ) -> float:
        """
        应用个体校准，返回校准后的预测值。
        当条件不满足时，直接返回全局预测。
        """
        # 激活条件检查
        if wear_days < 60 or labeled_sample_count < 5:
            return global_prediction

        bias = self._bias_store.get(dog_id, 0.0)

        # 混合权重：个体样本越多，偏信个体校准
        # 5条样本 → 30% 个体权重；20条以上 → 80% 个体权重
        individual_weight = min(0.8, 0.3 + (labeled_sample_count - 5) * 0.025)
        global_weight = 1.0 - individual_weight

        # 加权混合
        calibrated = global_weight * global_prediction + individual_weight * (global_prediction + bias)
        return float(np.clip(calibrated, 0.0, 10.0))
```

### 8.3 冷启动行为

| 阶段 | wear_days | 标签样本数 | 系统行为 |
|-----|----------|-----------|---------|
| 新用户（无数据） | 0–13 | 0 | 使用规则系统后备（XGBoost 不激活）|
| 过渡期 | 14–59 | 0–4 | 全局 XGBoost，无个体校准 |
| 个体校准激活 | ≥60 | ≥5 | 全局 XGBoost + 个体校准混合 |
| 高置信个体化 | ≥60 | ≥20 | 个体校准权重提升至 80% |

---

## 9. 实施时间线与里程碑

### 9.1 详细时间线

#### 第 1–2 周：特征存储管线搭建（无模型）

**目标**：在现有 `evaluator.py` 的 `assess_device()` 函数执行后，同时将特征数据写入 `pet_daily_ml_features` 表。

**任务清单**：
- [ ] 创建 `pet_daily_ml_features` 表（执行第2节 SQL）
- [ ] 在 `evaluator.py` 中添加 `_write_ml_features()` 异步函数，在 upsert 完成后调用
- [ ] 实现时序特征的计算逻辑（`compute_temporal_features()`）
- [ ] 添加数据质量监控：每日统计特征空值率，输出到日志
- [ ] 编写单元测试验证特征计算逻辑正确性

**验收标准**：运行7天后，`pet_daily_ml_features` 表有正确数据；所有非时序特征空值率 < 5%。

#### 月份 1–3：数据积累与标签收集

**目标**：积累足够的训练数据；启动问卷标签收集。

**任务清单**：
- [ ] App 端追问树问卷完成度监控（目标：活跃用户每周至少1次填写）
- [ ] 实现 `label_source` 和 `label_weight` 的自动更新逻辑
- [ ] 在第4–6周内发布 App 端"就诊记录"功能，开始收集 vet 标签
- [ ] 每两周运行一次数据质量报告，检查特征分布是否符合预期

**Go/No-Go**：
- ≥ 50 只犬积累 ≥ 14 天数据
- questionnaire_level 填写率 ≥ 30%（即30%的记录有问卷标签）
- 无系统性特征异常（例如所有犬的 z_peb 恒为0等管线bug）

#### 月份 3：第一版 XGBoost 训练（冷启动版本）

**目标**：使用规则系统标签（label_source="rule"）训练第一个可测试的模型。

**任务清单**：
- [ ] 运行 `train_and_save()`，使用 rule 标签作为目标变量
- [ ] 验证 CV MAE ≤ 1.5 级，within-1 准确率 ≥ 70%
- [ ] 开始 Phase 1 静默运行（XGBoost 在后台计算，不展示）
- [ ] 建立分歧监控仪表板

**注意**：此时 XGBoost 本质上是在学习规则系统的输出，预期性能不会超过规则系统，主要目的是验证管线端到端可用。

#### 月份 3–6：并行运行与弱标签训练

**目标**：用问卷弱标签重训模型，开始观察是否超越规则系统。

**任务清单**：
- [ ] 月份4初：用 questionnaire+rule 混合标签重训，观察 CV MAE 是否下降
- [ ] 启动 Phase 2（内部双轨展示），算法团队每周审查分歧案例
- [ ] 与医疗团队共同评估分歧案例，确定哪个系统更准确
- [ ] 建立"优先标注队列"：分歧 >= 2 级的案例自动推送给运营人员获取 vet 标签

**Go/No-Go 进入 Phase 3**：
- 在有问卷标签的记录上，XGBoost MAE < 规则系统 MAE（即 XGBoost 更接近主人观察）
- 至少有50条 vet 标签，XGBoost 在这些记录上 MAE ≤ 1.0 级
- 医疗团队审查分歧案例，确认 XGBoost 更合理的比例 ≥ 60%

#### 月份 6：切换为主评分系统

**目标**：Phase 3 上线，XGBoost 成为用户看到的评分系统。

**任务清单**：
- [ ] 实现 `should_use_rule_fallback()` 逻辑
- [ ] SHAP 解释文本写入数据库，App 端展示"主要原因"模块
- [ ] 设立回滚方案：一键恢复规则系统（通过配置开关，无需代码发布）
- [ ] 监控切换后2周的用户反馈和错误报告

#### 月份 12：个体化校准层

**目标**：对佩戴超过60天且有足够标签的犬只启用个体校准。

**任务清单**：
- [ ] 实现 `IndividualCalibrator` 的持久化（偏置值存入数据库，服务重启后恢复）
- [ ] 每周批量更新个体校准偏置
- [ ] A/B 测试：有个体校准 vs 无个体校准的 MAE 对比
- [ ] 评估是否将兽医标签纳入个体校准训练（避免过拟合单次诊断）

### 9.2 里程碑汇总

| 里程碑 | 时间 | 验收标准 |
|-------|-----|---------|
| M1: 特征存储就绪 | 第2周 | 数据正确写入，无系统性空值 |
| M2: 第一版模型 | 月份3 | CV MAE ≤ 1.5，并行运行开始 |
| M3: 超越规则系统 | 月份5 | 在问卷标签上 XGBoost MAE < 规则 MAE |
| M4: Phase 3 上线 | 月份6 | SHAP 解释上线，用户可见 |
| M5: 个体化 | 月份12 | 校准层 A/B 测试 MAE 改善 ≥ 10% |

---

## 10. 风险与应对

| 风险 | 概率 | 影响 | 缓解策略 |
|-----|-----|-----|---------|
| **标注数据严重不足** | 高 | 高 | ① 以规则系统输出做冷启动自监督；② 问卷弱标签权重0.6；③ 严重场景优先标注队列；④ 接受冷启动版本不超越规则系统，耐心积累 |
| **类别严重不平衡（L0-L2占80%）** | 高 | 中 | ① 反频率 sample_weight；② 有序回归目标对极端值天然容忍；③ 自定义非对称损失惩罚漏报；④ 监控 severe_recall 指标，而非仅看整体 MAE |
| **XGBoost 过拟合规则系统标签** | 中 | 高 | ① 尽早混入问卷弱标签（月份3即开始）；② GroupKFold 防止泄露；③ 用问卷标签而非规则标签做最终评估；④ 在分歧分析中主动寻找规则系统的系统性偏差 |
| **个体差异过大，全局模型失效** | 中 | 中 | ① 个体校准层（月份12）；② breed_encoded 和 age_months 作为特征帮助全局模型区分个体类型；③ Phase 3 的 fallback 机制保护新用户 |
| **用户不信任"黑盒"系统** | 低 | 高 | ① SHAP 解释文本是强制要求，非可选项；② SHAP 解释必须用用户能理解的语言（"抓挠频次异常"而非"z_peb=2.55"）；③ 同时展示规则系统参考值增加可信度 |
| **特征计算管线 Bug 导致特征失效** | 中 | 高 | ① 特征空值率监控告警；② `feature_version` 字段标记特征计算逻辑版本；③ 历史数据在特征逻辑改变时需要重新计算 |
| **主人问卷依从性低（<20%）** | 中 | 中 | ① 弱标签覆盖率不足时，自监督标签可以延长；② 研究最佳推送时机提升填写率；③ 接受冷启动版本主要依赖自监督 |

---

## 11. 与现有代码的集成点

### 11.1 在管线中的位置

XGBoost 评分层插入在 Step 4（特征提取）之后，替换 Step 5（日度评估计算）中的评分数学，**不修改** `modules/inference/model.py`（行为分类器）和 `modules/baseline/updater.py`（基线更新）的任何逻辑。

**现有代码调用链**（`scheduler/jobs.py` 触发）：
```
run_batch_assessment()
  → assess_device()           # modules/assessment/evaluator.py
      → [Step 1-4] 特征聚合
      → [Step 5] 计算 S1-S6 分数 → rule_score, rule_level   ← 此处插入 XGBoost
      → upsert pet_skin_health_daily
```

**插入点**：在 `assess_device()` 函数中，`alert_reason` 和规则系统 upsert 计算完毕后，调用 `ScoringModel.predict()`，将 XGBoost 预测结果写入 `pet_daily_ml_features`。

### 11.2 与 BehaviorClassifier 的类比

XGBoost 评分模型遵循与 `BehaviorClassifier`（`modules/inference/model.py`）相同的单例加载模式：

```python
# modules/scoring/model.py  （新增文件）
"""
XGBoost 评分模型 Wrapper。

架构模式与 BehaviorClassifier (modules/inference/model.py) 一致：
- 单例加载（服务启动时加载一次，推理时重用）
- 权重文件存于 weights/ 目录
- predict() 方法接受特征向量，返回预测等级和置信度
"""

import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import shap

from config import settings


@dataclass
class ScoringResult:
    level: int              # L0-L10（0-10整数）
    raw_score: float        # XGBoost 回归原始输出（浮点）
    confidence: float       # 置信度 [0, 1]
    shap_values: np.ndarray # 各特征 SHAP 贡献值
    model_version: str


class ScoringModel:
    """XGBoost 评分模型单例包装器。"""

    def __init__(self):
        path = Path(settings.scoring_model_path)  # 新增配置项: "weights/scoring_xgb_v1.pkl"
        if not path.exists():
            raise FileNotFoundError(f"评分模型未找到: {path}")

        with open(path, "rb") as f:
            artifact = pickle.load(f)

        self._model = artifact["model"]
        self._features = artifact["features"]
        self._model_version = artifact.get("model_version", "unknown")

        # TreeExplainer 在初始化时预计算树结构，推理时速度很快
        self._explainer = shap.TreeExplainer(self._model)

    @property
    def features(self) -> list[str]:
        return self._features

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, feature_vector: np.ndarray) -> ScoringResult:
        """
        feature_vector : (n_features,) float32，顺序必须与 self.features 一致
        """
        X = feature_vector.reshape(1, -1).astype(np.float32)

        raw_score = float(self._model.predict(X)[0])
        level = int(np.clip(round(raw_score), 0, 10))

        # 置信度：预测值距最近整数边界的距离
        dist_to_boundary = abs(raw_score - round(raw_score))
        confidence = float(np.clip(0.5 + dist_to_boundary * 1.5, 0.5, 0.95))

        # SHAP 值（每次预测都计算，性能影响约1ms/次，可接受）
        shap_vals = self._explainer.shap_values(X)[0]

        return ScoringResult(
            level=level,
            raw_score=round(raw_score, 3),
            confidence=round(confidence, 3),
            shap_values=shap_vals,
            model_version=self._model_version,
        )

    def is_available(self) -> bool:
        """检查模型是否已加载（用于 fallback 逻辑）。"""
        return self._model is not None


# 单例 — 与 BehaviorClassifier 保持相同模式
_scoring_model: ScoringModel | None = None


def get_scoring_model() -> ScoringModel | None:
    """
    获取评分模型单例。若权重文件不存在则返回 None（不抛出异常），
    调用方负责处理 None 的情况（退回规则系统）。
    """
    global _scoring_model
    if _scoring_model is None:
        try:
            _scoring_model = ScoringModel()
        except FileNotFoundError:
            return None  # 模型尚未训练，正常情况
    return _scoring_model
```

### 11.3 weights/ 目录规范

与现有的 `weights/behavior_lgbm.pkl` 并列存放：

```
weights/
├── behavior_lgbm.pkl          # 行为分类模型（现有）
├── scoring_xgb_v1.pkl         # XGBoost 评分模型 v1（新增）
├── scoring_xgb_v2.pkl         # 未来版本（灰度切换）
└── .gitkeep
```

**版本管理规则**：
- 每次重训产生带版本号的文件：`scoring_xgb_v{MAJOR}.{MINOR}.{PATCH}-{DATE}.pkl`
- `settings.scoring_model_path` 指向当前生产版本
- 灰度切换时通过修改配置实现，无需代码发布

### 11.4 config.py 新增配置项

在 `config.py` 的 `Settings` 类中添加：

```python
# XGBoost 评分模型
scoring_model_path: str = "weights/scoring_xgb_v1.pkl"

# 并行运行控制
scoring_use_xgb: bool = False         # False=规则系统, True=XGBoost主导
scoring_xgb_silent: bool = True       # True=静默运行不展示, False=展示给用户
scoring_fallback_min_wear_days: int = 14   # 少于此天数退回规则系统
scoring_fallback_min_confidence: float = 0.60  # 低于此置信度退回规则系统
```

### 11.5 在 evaluator.py 中的调用方式

在 `assess_device()` 函数末尾（规则系统 upsert 完成后）添加：

```python
# ── 6. XGBoost 并行评分（不阻塞主流程，失败时静默跳过）────────────────
try:
    from modules.scoring.model import get_scoring_model
    from modules.scoring.features import build_feature_vector  # 新增工具函数

    scoring_model = get_scoring_model()
    if scoring_model is not None:
        # 构建特征向量（从本次评估数据 + 历史查询）
        feature_vec = await build_feature_vector(
            db=db,
            device_sn=device_sn,
            stat_date_ts=stat_date_ts,
            current_day_data={
                "zscore": zscore,
                "rule_score": rule_score_value,  # 规则系统输出
                "rule_level": rule_level_value,
                # ... 其他当日数据
            },
        )
        result = scoring_model.predict(feature_vec)

        # 判断是否使用 XGBoost 结果（Phase 决策）
        use_xgb, _ = should_use_rule_fallback(
            {"wear_days": valid_days, ...},
            result.confidence,
        )

        # 写入 ML 特征表（无论是否使用 XGBoost，特征数据都要存储）
        await _write_ml_features(
            db=db,
            device_sn=device_sn,
            stat_date_ts=stat_date_ts,
            feature_vec=feature_vec,
            xgb_result=result,
            rule_score=rule_score_value,
            rule_level=rule_level_value,
        )

except Exception:
    logger.exception(
        "XGBoost 评分失败（静默跳过）device=%s stat_date_ts=%d",
        device_sn, stat_date_ts,
    )
    # 不重新抛出异常，XGBoost 失败不影响规则系统正常运行
```

---

## 附录 A：特征列顺序与推理时的严格一致性

**警告**：XGBoost 模型推理时，特征顺序必须与训练时完全一致。`FEATURES` 列表定义在 `xgboost_trainer.py` 中，推理时的 `ScoringModel` 从 pkl 文件加载同一份 `features` 列表。任何新增特征必须同时更新两处，并触发重新训练。

```python
# 训练时
artifact = {"model": model, "features": FEATURES, ...}  # 保存特征列表

# 推理时
self._features = artifact["features"]  # 从保存的列表恢复，确保一致性
```

## 附录 B：模型文件审计

每个 pkl 文件包含以下信息，可随时检查：

```python
with open("weights/scoring_xgb_v1.pkl", "rb") as f:
    artifact = pickle.load(f)

print(artifact["model_version"])       # 版本号
print(artifact["training_samples"])    # 训练样本数
print(artifact["training_dogs"])       # 训练犬只数
print(artifact["cv_metrics"])          # CV 评估指标
print(artifact["label_distribution"])  # 训练集标签分布
print(artifact["features"])            # 特征列表（推理时必须一致）
```

---

*本文档为算法工程团队内部设计文档，涉及模型训练策略、数据标注规范和系统集成方案。*  
*文档维护：算法工程团队*  
*最后更新：2026-05-25*
