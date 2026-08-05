# 🌾 灌溉需求预测

> Kaggle Playground Series S6E4 - 从 0.96 到 0.98153 的完整竞赛迭代

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E4-20BEFF.svg)](https://www.kaggle.com/competitions/playground-series-s6e4)
[![Score](https://img.shields.io/badge/Private%20LB-0.98153-brightgreen.svg)]()

## 📊 项目概览

| 项目 | 内容 |
|------|------|
| 比赛 | Kaggle Playground Series Season 6, Episode 4 |
| 任务 | 三分类：预测农田灌溉需求等级（Low / Medium / High）|
| 数据规模 | 训练集 900,000 条（官方 63 万 + 额外 27 万），测试集 270,000 条 |
| 评估指标 | Balanced Accuracy（平衡准确率）|
| 最终成绩 | **Private Score: 0.98149** / Public Score: 0.98152 |
| 最终排位 | **223 / 4316** / 64 / 4316 |
| 排名 | Top 5% |

![alt text](<images/Public LB截图.png>)
![alt text](<images/private LB截图.png>)

### 目标分布（类别严重不平衡）

| 类别 | 占比 |
|------|------|
| Low | 71.10% |
| Medium | 26.56% |
| High | 2.33% |

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据流全景图                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📂 原始数据 (90万条)                                                       │
│       │                                                                     │
│       ▼                                                                     │
│  🔧 特征工程                                                                │
│  ├── 数字根特征 (digit -4~4)  ──────────────────► 捕捉数值细粒度模式         │
│  ├── 交互特征 (类别×类别组合) ──────────────────► 捕捉特征组合信息           │
│  ├── Logit 分数特征           ──────────────────► 注入农业领域知识           │
│  └── 阈值二值特征             ──────────────────► 简化关键决策边界           │
│       │                                                                     │
│       ▼                                                                     │
│  🤖 模型训练 (XGBoost)                                                     │
│  ├── 5折交叉验证                                                            │
│  ├── Optuna 超参数调优                                                      │
│  └── 类别权重处理 (处理不平衡)                                              │
│       │                                                                     │
│       ▼                                                                     │
│  🏷️ 伪标签 (Pseudo-Labeling)                                              │
│  └── 置信度阈值 0.95 → 选取 ~25万 高置信度样本扩充训练集                    │
│       │                                                                     │
│       ▼                                                                     │
│  🔗 投票集成 (6个基模型)                                                    │
│  └── 密码串 (microEDA) 分歧诊断 + 规则投票                                  │
│       │                                                                     │
│       ▼                                                                     │
│  📤 提交文件 (Private LB: 0.98153)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 核心突破

### 1️⃣ 数字根特征（贡献约 +0.01）

将数值特征的每一位数字拆分为独立特征，让模型获得多尺度视角。

```python
def add_digit_features(df, numeric_cols):
    for c in numeric_cols:
        for k in range(-4, 5):
            # 提取第 k 位数字
            df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('int8')
    return df
```

**示例**：`Soil_Moisture = 23.7`
- `digit1` = 2（十位）→ 大范围湿度级别
- `digit0` = 3（个位）→ 中范围湿度级别
- `digit-1` = 7（十分位）→ 精细湿度级别

### 2️⃣ 伪标签（贡献约 +0.006）

用测试集高置信度预测（>95%）扩充训练集，实现半监督学习。

```python
# 选取高置信度样本
mask = test_probs.max(axis=1) >= 0.95

# 扩充训练集
X_aug = pd.concat([X_full, X_test_f[mask]])
y_aug = np.concatenate([y, test_probs[mask].argmax(axis=1)])

# 重新训练
final_model.fit(X_aug, y_aug)
```

### 3️⃣ 交互特征（贡献约 +0.005）

生成类别特征的交叉组合，捕捉组合规律。

```python
for c1, c2 in combinations(cat_cols, 2):
    combined = df[c1].astype(str) + "_" + df[c2].astype(str)
    # 编码为整数特征
```

**示例**：`Soil_Type` + `Crop_Growth_Stage` 的组合可能比单独使用更有效。

### 4️⃣ 密码串投票集成

将多模型预测压缩为"密码串"，针对分歧模式精确调优。

```python
def microEDA(x, list_names_df):
    # 将 6 个模型的预测压缩为密码串
    # 例如：'L L L _ _ H ' 表示 3 个 Low、2 个 Medium、1 个 High
    lwh = []
    for _p in list_names_df:
        wh = x[_p][0:1]  # L / M / H
        if wh == 'M': wh = '_'  # Medium 用下划线代替
        lwh.append(f'{wh} ')
    return ''.join(lwh)
```

**分歧热力图**：用于诊断哪些模型在哪些类别上容易出错。

---

## 📈 版本迭代全记录

| 版本 | 核心策略 | CV 分数 | 公开分数 | 状态 |
|------|---------|---------|---------|------|
| **V1** | XGBoost + Optuna 调参 + 基础特征 | ~0.976 | - | ✅ 基线 |
| **V2** | XGBoost + LightGBM 集成 | ~0.962 | - | ❌ 特征太简单 |
| **V3** | 三模型集成（XGB+LGB+CatBoost）| ~0.962 | - | ❌ 无提升 |
| **V4** | LightGBM + 数字根特征 | 0.9739 | **0.97305** | 🏆 银牌方案 |
| **V5** | 扩展特征 + 多种子 + Optuna后处理 | - | **0.93** | ❌ 严重过拟合 |
| **V6** | XGBoost + LightGBM 加权集成 | 0.8597 | 0.968 | ❌ 代码 bug |
| **V7** | XGBoost + 交互特征 + 伪标签 | 0.9728 | **0.978** | 🏆 突破 |
| **V8** | 多模型 + SMOTE + PCA + Stacking | - | **0.968** | ❌ 过度工程 |
| **V9** | V7 简化版（5折 + 高置信度伪标签）| 0.97299 | **0.97943** | 🏆 最佳单模型 |
| **V10** | OrderedTE + XGBoost Trio | 报错 | - | ❌ 实验失败 |
| **V11-V14** | 多模型集成优化（6个基模型）| - | **0.98152-0.98153** | 🏆 冲刺 |
| **V15-V24** | 投票规则迭代（约10次）| - | 0.98152-0.98153 | ⚠️ 原地踏步 |
| **V25** | 引入第 7 票（公式特征）| - | **0.98145** | ❌ 过拟合 |

### 📊 版本演进可视化

```
分数
0.982 ┤
      │                                    ● V24 (0.98153)
0.981 ┤                              ●─────●─────●
      │                            ●  V14  V20  V22
0.980 ┤                      ● V9
      │                    ●
0.979 ┤                  ●
      │                ●
0.978 ┤          ● V7
      │         ●
0.977 ┤       ●
      │      ●
0.976 ┤     ● V1
      │    ●
0.975 ┤   ●
      │  ●
0.974 ┤ ● V4 (0.97305)
      │●
0.973 ┤─────────────────────────────────────────────────
      │
      └────┬────┬────┬────┬────┬────┬────┬────┬────┬────
          V1   V4   V7   V9   V14  V20  V24  V25
              银牌       突破  冲刺       过拟合
```

---

## 💡 踩坑经验总结

### ❌ 教训一：特征爆炸（V5：0.973 → 0.93）

**问题**：特征数从 86 暴增到 267，模型严重过拟合

**原因**：
- 扩展数字根到 ±6 没有理论依据，引入了大量噪声
- 交叉统计特征在测试集上分布可能不同
- 多种子训练 + 后处理优化加剧了过拟合

**启示**：**特征不是越多越好。每个特征都应该有明确的设计理由。**

---

### ❌ 教训二：过度工程（V8：0.978 → 0.968）

**问题**：SMOTE + PCA + Stacking 导致性能全面下降

**原因**：
- SMOTE 生成的人造样本可能不符合真实分布（High 类只有 2.3%）
- PCA 丢失了特征的具体含义
- Stacking 的元模型（LogisticRegression）太弱，无法有效利用基模型输出

**启示**：**当特征已经足够丰富时，简单的方法往往更有效。**

---

### ❌ 教训三：Public LB 过拟合（V15-V25）

**问题**：10 多次投票规则调参，分数在 0.98152-0.98153 之间震荡

**原因**：
- 每个"补丁"都是针对 Public LB 的失败模式设计的
- Public LB 只包含 30% 的测试数据，不代表完整分布
- V25 引入的第 7 票（公式特征）质量太差（BA 仅 0.96），污染了投票池

**启示**：**在 0.98 以上的区间，需要的是更好的基模型，而不是更复杂的投票规则。**

---

## 📁 文件结构

```
Irrigation-Need-Prediction/
├── README.md                         # 项目说明（本文件）
├── .gitignore                        # Git 忽略规则
├── 预测灌溉.ipynb                    # 主代码（V1-V10 完整迭代）
├── Ensemble of solutions.ipynb       # V11-V25 投票集成实验
│
├── images/                           # 可视化图片
│   ├── feature_importance.png        # 特征重要性
│   └── target_distribution.png       # 目标分布
│
└── submissions/                      # 代表性提交文件
    ├── submission_lgb_silver.csv     # V4 银牌方案 (0.97305)
    ├── submission-v7.csv             # V7 伪标签方案 (0.978)
    └── submission_optimized.csv      # V9 最佳单模型 (0.97943)
```

---

## 🛠️ 技术栈

| 类别 | 工具/库 |
|------|---------|
| 语言 | Python 3.10 |
| 数据处理 | Pandas, NumPy |
| 机器学习 | Scikit-learn, XGBoost, LightGBM |
| 超参数调优 | Optuna |
| 可视化 | Matplotlib, Seaborn |
| 版本控制 | Git, GitHub |

---

## 🚀 快速复现

```bash
# 1. 克隆仓库
git clone https://github.com/hehaha644329615/Irrigation-Need-Prediction.git
cd Irrigation-Need-Prediction

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据（从 Kaggle 比赛页面）
# 将 train.csv, test.csv, sample_submission.csv 放入项目根目录

# 4. 运行主代码
jupyter notebook 预测灌溉.ipynb
```

---

## 📊 关键结果

| 指标 | 值 |
|------|-----|
| 最终 Private Score | **0.98153** |
| 最终 Public Score | 0.98152 |
| 排名 | Top 5% |
| 最优 CV 分数 | 0.976 |
| 特征数量 | ~150（最终版本）|
| 基模型数量 | 6 |
| 伪标签样本数 | ~25万 |

---

## 📝 致谢与反思

### 收获
1. **特征工程 > 模型调参**：数字根特征的贡献远超 Optuna 调参
2. **伪标签是宝藏**：充分利用测试集可以显著提升泛化能力
3. **集成需要克制**：简单平均往往比复杂 Stacking 更好
4. **Public LB 是陷阱**：不要针对 Public LB 调参

### 可改进方向
- 尝试更精细的领域知识特征
- 使用 GPU 训练更深的树模型
- 探索 LightGBM 的多种子训练

---

## 🔗 相关链接

- [Kaggle 比赛页面](https://www.kaggle.com/competitions/playground-series-s6e4)
- [我的 Kaggle 个人页面](https://www.kaggle.com/hehaha644329615)

---

<div align="center">
  <sub>Built with ❤️ by hehaha644329615</sub>
</div>