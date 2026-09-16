# acc3_rf —— 只用加速计的模型（5 通道 113 维）

放 `ml_rf.pkl` + `ml_rf.json` 进这个目录，然后把 `MODEL_PATH` 指过来，
线上这条链就跑只用加速计的版本。**默认还是 `weights/stable_v2_rf/`，这里是给对比用的。**

陀螺仪很费电。这个模型用来量：**砍掉它，线上这条链的输出差多少。**

## 装

在训练机上训好之后（见 `imu_train/acc3/README.md`）：

```bash
scp ~/imu_train/results_acc3/*_acc3/16hz_remap_custom_3class/rf/ml_rf.{pkl,json} \
    <这台>:~/algo_service/weights/acc3_rf/
```

## 切过去

```bash
# docker-compose.yml 里加环境变量，或者 .env 里：
MODEL_PATH=weights/acc3_rf/ml_rf.pkl

cd ~/algo_service && docker compose down && docker compose up -d --build
```

启动日志里会多一行，**确认它真的在跑取列那条路**：

```
特征取列 : 从 193 维里取 113 列（只用加速计，陀螺仪那些维不参与）
```

没有这一行就是还在跑 `stable_v2_rf`。

## 为什么不用改别的配置

模型是 5 通道（acc 三轴 + pitch/roll）113 维的，而这边的特征提取固定按
8 通道算、出来 193 维。但那 113 维**正好是 193 维里去掉所有陀螺仪相关维度
之后的那些，顺序一致、数值逐位相同**——所以照常算 193 维，再按模型自己
`ml_rf.json` 里的 `feature_select` 取列就行，不需要另一条预处理链。

下标存在模型的 json 里，不是这边按特征名现筛的：这边的
`modules/inference/features.py` 跟训练机那份 `imu_train` 可能不是同一版，
现筛出来的下标会**整体错位**，而错位**不报错**——每一维都对到别的特征上，
模型照样给得出概率。所以 json 里连 `from_dim`（193）一起存，
这边算出来不是 193 维就当场报错。

## ⚠ 换模型会改变写进库的数

这个模型跟 `stable_v2_rf` 的类别、几何（16Hz / 1s 窗 / 0.5s 步）完全一样，
所以 `pet_dog_behavior` / `pet_dog_daily_summary` 的**表结构不用动**。
但**数值会变**——离线验证集上（同一批窗口，唯一差别是有没有陀螺仪）：

| 类别 | stable_v2_rf | acc3_rf | 差 |
|---|---|---|---|
| 活动 | 0.860 | 0.766 | −0.094 |
| 抓挠 | 0.769 | 0.699 | −0.070 |
| 睡觉 | 0.902 | 0.940 | +0.038 |
| 未佩戴 | 0.895 | 0.951 | +0.057 |

**动的那两类掉，静止那两类涨。** 所以切过去之后日汇总里的
「活动时长」会变短、「睡觉时长」会变长，**这不是 bug**。
要对比就把两边的日汇总分开存/分开导，别混着看。
