```yaml
---
title: "SEER癌症患者生存月数预测"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/seer-survival-prediction
date: 2026-01-01
excerpt: "基于SEER癌症数据集，使用决策树、随机森林与SVM模型预测患者生存月数，对比不同模型性能差异"
header:
  teaser: /images/portfolio/seer-survival-prediction/cover.png
tags:
- 机器学习
- 生存分析
- 医疗数据
- 预测建模
tech_stack:
- name: Python
- name: Scikit-learn
- name: Pandas
- name: Matplotlib
---

## 项目背景

SEER（Surveillance, Epidemiology, and End Results）数据库是美国国家癌症研究所（NCI）建立的大型癌症监测数据库，包含了大量癌症患者的临床信息和随访数据。本项目利用SEER数据集，构建机器学习模型来预测癌症患者的生存月数，为医疗决策提供数据支持。

## 核心实现

### 数据预处理

首先对原始数据进行清洗和编码处理：
```python
# 去除缺失值
data = data.dropna()

# 分类变量标签编码
label_encode_columns = [col for col in data.columns if col not in ['Survival months', 'Year of diagnosis', 'Year of follow-up recode', 'Behavior code ICD-O-3']]
for col in label_encode_columns:
    data[col] = data[col].astype(str)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col]) + 1
    label_encoders[col] = le
```

### 模型构建

本项目对比了三种不同的机器学习模型：

#### 决策树模型
```python
model = Pipeline(steps=[
    ('regressor', DecisionTreeRegressor(
        criterion='friedman_mse',
        splitter='best',
        max_depth=10,
        max_leaf_nodes=50,
        random_state=42
    ))
])
```

#### 随机森林模型
```python
model = Pipeline(steps=[
    ('regressor', RandomForestRegressor(
        criterion='squared_error',
        max_features=None,
        max_depth=10,
        max_leaf_nodes=50,
        n_estimators=100,
        random_state=42
    ))
])
```

#### SVM模型
```python
model = Pipeline(steps=[
    ('regressor', SVR(
        C=1,
        kernel='linear',
        gamma='scale',
        max_iter=1000
    ))
])
```

## 分析结果

### 模型性能对比

三个模型在测试集上的性能指标如下：

| 指标 | 决策树 | 随机森林 | SVM |
| :--- | :--- | :--- | :--- |
| MSE | 559.84 | 182.46 | 572.80 |
| RMSE | 23.66 | 13.51 | 23.93 |
| MAE | 13.33 | 9.14 | 20.18 |
| R² | 0.81 | 0.82 | 0.95 |

### 可视化结果

#### 决策树模型预测结果

![决策树模型预测对比](/seer-survival-prediction/dt_true_vs_pred.png)

决策树模型对前1000个样本的预测值与真实值对比，模型能够捕捉到数据的整体趋势，但在细节拟合上存在一定误差。

#### 随机森林模型预测结果

![随机森林模型预测对比](/seer-survival-prediction/rf_true_vs_pred.png)

随机森林模型通过集成多个决策树，显著提升了预测精度，预测曲线更加贴近真实值。

#### SVM模型预测结果

![SVM模型预测对比](/seer-survival-prediction/svm_true_vs_pred.png)

SVM模型在处理非线性关系上表现较好，但收敛速度较慢，在大数据集上的训练效率较低。

## 项目总结

本项目通过对比三种机器学习模型在SEER数据集上的表现，验证了随机森林模型在医疗生存分析任务中的优势。同时，展示了完整的医疗数据建模流程，包括数据预处理、模型构建、性能评估和可视化分析，为类似项目提供了可参考的实践框架。
