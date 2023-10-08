from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
"""
1. cross validation
把训练集进一步分成训练和验证集
如把训练集分成四份，其中一份作为验证集得到模型准确率，然后更换验证集，
先后进行四组测试，得到4组模型的准确率，取平均值为最终结果，
即4折交叉验证

2. 超参数搜索 - 网格搜索（grid search）
2.1 很多参数需要手动指定（如knn算法中的k值），这样的参数被称为超参数
2.2 网格搜索简化了手动遍历各超参数的过程
需要对模型预设几种超参数组合，每组超参数都采用交叉验证来进行评估，最后选出最优参数建立模型
"""

"""
sklearn.model_selection.GridSearchCV(estimator, param_grid=None, cv=None)
1. estimator: 估计值对象
2. param_grid: 估计器参数(dict) {"n_neighbors": [1,3,5]}
（n_neighbors是对于KNN来说，此处填写需求的超参数）
3. cv: 指定几折交叉验证
· 实例化后使用方法：
fit(): 输入训练数据
score(): 准确率
· 结果分析：
最佳参数：best_param_
最佳结果：best_score_
最佳估计器：best_estimator_
交叉验证结果：cv_results_
"""
# Add cross validation and grid-search to the iris case

# 1. 获取数据
iris = datasets.load_iris()

# 先划分数据集后特征工程，标准化统一使用训练集数据，测试集也使用训练集参数
# 2. 数据集划分
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# 3. 特征工程标准化
# test集使用transform方法：使用的是上一步train集的参数直接计算
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. KNN预估器流程
estimator = KNeighborsClassifier()

# 加入网格搜索和交叉验证
param_dict = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
estimator = GridSearchCV(estimator=estimator, param_grid=param_dict)
estimator.fit(x_train, y_train)

# 5. 模型评估
# 方法一：predict()后直接比对
y_predict = estimator.predict(x_test)
print("y_predict: \n", y_predict)
print("直接比对真实值和预测值: \n", y_test == y_predict)

# 方法二：score()计算准确率
accuracy = estimator.score(x_test, y_test)
print("准确率: \n", accuracy)

# 最佳参数：best_param_
print("最佳参数: \n", estimator.best_params_)
# 最佳结果：best_score_
print("最佳结果: \n", estimator.best_score_)
# 最佳估计器：best_estimator_
print("最佳估计器: \n", estimator.best_estimator_)
# 交叉验证结果：cv_results_
print("交叉验证结果: \n", estimator.cv_results_)