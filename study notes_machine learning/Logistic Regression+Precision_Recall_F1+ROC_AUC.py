"""
Logistic Regression API:
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
    1. solver: 优化求解方式
    默认开源的liblinear库实现，内部使用坐标轴下降法来迭代优化损失函数
    2. penalty: 正则化种类
    3. C: 正则化力度

LogisticRegression方法相当于SGDClassifier(loss='log', penalty=''),
SGDClassifier实现普通的随机梯度下降学习，也可以通过设置average=True实现平均随机梯度下降法SAG
LogisticRegression实现SAG
"""
"""
使用logistic回归对癌症进行分类
1. 获取数据，对数据添加columns
2. 数据处理，缺失值处理（在该数据集中无缺失值）
3. 划分数据集
4. 特征工程：无量纲化（标准化）
5. 逻辑回归预估器
6. 模型评估
"""
# 1. 获取数据，对数据添加columns
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

cancer_ori = load_breast_cancer()
cancer = pd.DataFrame(cancer_ori.data, columns=cancer_ori.feature_names)

# 2. 数据缺失值处理
# 1）替换？为np.nan
cancer = cancer.replace(to_replace="?", value=np.nan)

# 2）删除缺失样本
cancer.dropna(how="any", inplace=True)
print(cancer.head())

# 3. 划分数据集
x = cancer
y = cancer_ori.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x, y)

# 4. 特征工程标准化
from sklearn.preprocessing import StandardScaler
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 5. 逻辑回归预估器
from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
estimator.fit(x_train, y_train)
weight = estimator.coef_
bias = estimator.intercept_
print("weight = ", weight, "bias = ", bias)

# 6. 模型评估
y_predict = estimator.predict(x_test)
print("y_predict: \n", y_predict)
print("Direct Comparison: \n", y_predict == y_test)

score = estimator.score(x_test, y_test)
print("score: \n", score)

"""
Logistic Model Evaluation：Precision/Recall/F1
四种结果-混淆矩阵：
真实正例&预测正例：真正例TP True Positive
真实假例&预测正例：伪正例FP False Positive
真实正例&预测假例：伪反例FN False Negative
真实假例&预测假例：真反例TN True Negative

1. 精确率Precision：预测为正例的样本中真实结果为正例的比例
TP / (TP + FP)

2. 召回率Recall：真实为正例的样本中预测结果为正例的比例（查得全不全）
TP / (TP + FN)

3. F1-score：反映模型的稳健型
F1 = 2TP / (2TP + FN + FP) = 2 * Precision * Recall / (Precision + Recall)

API：
sklearn.metrics.classification_report(y_true, y_pred, labels=[], target_names=None)
y_true: 真实目标值
y_pred: 估计器预测目标值
labels: 类别对应的数字
target_names: 类别对应的名称
return: 每个类别的精确率/召回率/F1
"""
# 用参数评估癌症模型
from sklearn.metrics import classification_report
report = classification_report(y_test, y_predict, labels=[0, 1], target_names=["无癌症", "有癌症"])
print(report)

"""
当数据不对称时，上述参数不能很好地评估模型，为解决该问题，引入--
ROC & AUC
1. TPR / FPR
    TPR = TP / (TP + FN) - 召回率
        所有真实类别为1的样本中，预测类别为1的比例
    FDR = FP / (FP + TN) 
        所有真实类别为0的样本中，预测类别为1的比例
2. 横轴FPR 纵轴TPR的坐标系下
每个模型在某一阈值（判断类别的界限）下可计算确定的TPR/FPR并绘制一点，改变阈值，每个模型可以在该坐标系下画出一条曲线，即ROC曲线
AUC指标即该曲线右下部分面积大小，越接近0.5说明模型越不好（不负责任），越接近1说明越好
AUC closer to 1: more responsible model
AUC closer to 0.5: less responsible model

AUC API
sklearn.metrics.roc_auc_score(y_true, y_score)
    y_true: 每个样本的真实类别，必须以0（反例）/1（正例）标记
    y_score: 预测得分，可以是正类的估计概率、置信值或者分类器方法的返回值
"""
from sklearn.metrics import roc_auc_score
AUC = roc_auc_score(y_test, y_predict)
print("AUC: \n", AUC)