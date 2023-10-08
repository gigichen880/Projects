"""
1. 集成学习方法：由多个独立学习方法组成，每个方法返回一个预测值，最后取众数
2. Random Forest：森林，即包含多个决策树的分类器
3. 随机森林原理过程
训练集 N个样本 / M个特征
1）Random Training Sets - N个样本中随机有放回地抽样N个（bootstrap）
2）Random Features - 从M个特征中随机抽取m个特征，其中M>>m，有效降维

4. API
sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, bootstrap=True, random
-state=None, min_samples_split=2)
1) n_estimators: 森林中的树木数量，默认10
2) criteria: 同决策树 默认gini
3) max_depth: 同决策树 默认None
4) max_features='auto' 每个决策树最大特征数量
    if 'auto', then max_features=sqrt(n_features)
    if 'sqrt', same as 'auto'
    if 'log2', then max_feature=log2(n_features)
    if None, then max_features=n_features
5) bootstrap: 是否在构建树时使用放回抽样 默认True
6) min_samples_split: 节点划分最小样本数
7) min_samples_leaf: 叶子节点的最小样本数
超参数：n_estimator, max_depth, min_samples_split, min_samples_leaf
"""

# Predict survival in titanic
# 1. 获取数据
import pandas as pd
titanic = pd.read_csv("titanic.csv")

# 2. 提取特征值、目标值
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

# 3. 数据处理
# 1）缺失值处理
x['age'].fillna(x['age'].mean(), inplace=True)

# 2）转化成字典
x = x.to_dict(orient='records')
print(x)

# 4. 划分数据集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 5. 特征工程：字典特征抽取
from sklearn.feature_extraction import DictVectorizer
transfer = DictVectorizer(sparse=False)
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
print(x_train)
print(transfer.get_feature_names_out())

# 6. 决策树预估器流程
from sklearn.ensemble import RandomForestClassifier
estimator = RandomForestClassifier()
# 加入网格搜索和交叉验证 - 决定max_depth
from sklearn.model_selection import GridSearchCV
param_dict = {"n_estimators": [120, 200, 300, 500, 80, 1200],'max_depth': [5, 8, 15, 25, 30]}
estimator = GridSearchCV(estimator, param_grid=param_dict)
estimator.fit(x_train, y_train)

# 7. 模型评估
# 1) Direct Comparison
y_predict = estimator.predict(x_test)
print("y_predict: \n", y_predict)
print("直接比对：\n", y_predict == y_test)

# 2) Score
score = estimator.score(x_test, y_test)
print("准确率：\n", score)

