"""
Solution to overfitting：regularization
    1）L1：惩罚项 绝对值求和
    加上L1正则化的线性回归：LASSO
    2）L2: 惩罚项 平方求和
    加上L2正则化的线性回归：Ridge Regression
API：
sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, solver='auto', normalize=False)
    1）alpha: 正则化力度，lambda
    2）solver: 根据数据自动选择优化方法
    · 优化方法：
        1. GD Gradient Descent：
        原始的梯度下降法，每次迭代需要所有样本参与计算，计算量大，需要改进
        2. SGD Stochastic Gradient Descent
        随机梯度下降：每次迭代时随机抽取一个训练样本，用其数据更新梯度
        · 高效
        · 需要许多超参数：正则化参数、迭代数
        · 对于特征标准化敏感
        3. SAG Stochastic Average Gradient
        随机平均梯度法：每个样本保有旧梯度，每次随机一个样本更新梯度，然后到样本的梯度集中计算平均值来更新
    如果数据集、特征都比较大，选择SAG
    3）normalize: 数据是否进行标准化
    如果设置True，等同于在fit前调用preprocessing.StandardScaler标准化数据
    Ridge.coef_ 回归权重
    Ridge.intercept_ 回归偏置
注：Ridge方法相当于SGDRegressor(penalty='l2', loss='squared_loss'), 只不过SGDRegressor实现普通随机梯度下降GD，而Ridge实现SAG
Cross Validation in Ridge Regression
sklearn.linear_model.RidgeCV(_BaseRidgeCV, RegressorMixin)
    ·具有l2正则化的线性回归，可以进行交叉验证
    ·coef_：回归系数
"""
# Predict Boston house price

# 1）导入数据
import pandas as pd
boston = pd.read_csv("boston.csv")

# 2）划分数据集测试集
from sklearn.model_selection import train_test_split
x = boston.iloc[:, 0: -1]
y = boston["MEDV"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=17)

# 3）特征工程 - 无量纲化处理
from sklearn.preprocessing import StandardScaler
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4）预估器
from sklearn.linear_model import Ridge
estimator = Ridge()
estimator.fit(x_train, y_train)

# 5）得到模型
w = estimator.coef_
b = estimator.intercept_
print("w=", w, " b=", b)

# 6）回归性能评估 - 均方误差MSE
# sklearn.metrics.mean_squared_error(y_true, y_pred)
#   1. y_true: 真实值
#   2. y_pred: 预测值
y_predict = estimator.predict(x_test)
from sklearn.metrics import mean_squared_error
result = mean_squared_error(y_test, y_predict)

print("岭回归预测结果：\n", result)

