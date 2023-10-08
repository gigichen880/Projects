"""
Linear Regression
API：
1. Normal Equation
sklearn.linear_model.LinearRegression(fit_intercept=True)
    - fit_intercept: 是否计算偏置，默认True
    - LinearRegression.coef_可调用回归系数
    - LinearRegression.intercept_可调用偏置

2. Gradient Descent
sklearn.linear_model.SGDRegressor(loss="squared_error', fit_intercept=True,
learning_rate='invscaling', eta0=0.01)
    - loss="squared_error": 普通最小二乘法
    - fit_intercept: 是否计算偏置
    - learning rate: 学习率
        1）'constant': eta = eta0 (初始学习率)
        2) 'optimal': eta = 1.0 / (alpha * (t + t0))
        3) 'invscaling': eta = eta0 / pow(t, power_t)
            power_t = 0.25
    - SGDRegressor.coef_ 可调用回归系数
    - SGDRegressor.intercept_ 可调用偏置
"""
# Predict Boston House Price

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
from sklearn.linear_model import LinearRegression, SGDRegressor
estimator1 = LinearRegression()
estimator1.fit(x_train, y_train)

estimator2 = SGDRegressor(loss="squared_error", fit_intercept=True, learning_rate='invscaling', eta0=0.01)
estimator2.fit(x_train, y_train)

# 5）得到模型
w1 = estimator1.coef_
b1 = estimator1.intercept_
print("w1=", w1, " b1=", b1)

w2 = estimator2.coef_
b2 = estimator2.intercept_
print("w2=", w2, " b2=", b2)

# 6）回归性能评估 - 均方误差MSE
# sklearn.metrics.mean_squared_error(y_true, y_pred)
#   1. y_true: 真实值
#   2. y_pred: 预测值
y_predict1 = estimator1.predict(x_test)
y_predict2 = estimator2.predict(x_test)
from sklearn.metrics import mean_squared_error
result1 = mean_squared_error(y_test, y_predict1)
result2 = mean_squared_error(y_test, y_predict2)

print("回归方程预测结果：\n", result1)
print("梯度下降预测结果：\n", result2)

