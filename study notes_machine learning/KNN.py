from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""
1. Transformer - 特征工程的父类
1.1 Instantiate a transformer（实例化的是一个转换器类Transformer）
1.2 fit_transform
以标准化为例：
(x-mean) / std
fit_transform()
    fit() - 输入数据
    transform() - (x-mean) / std 进行最终转换（计算）

2. Estimator（sklearn机器学习算法的实现）
2.1 Instantiate an estimator
2.2 estimator.fit(x_train, y_train)计算
    - 调用完毕，模型生成
2.3 Evaluate Model
    1）直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    y_test == y_predict
    2) 计算准确率
    accuracy = estimator.score(x_test, y_test)
"""
"""
k nearest neighbor algorithm: 通过k个临近数据推断类别
计算距离：欧氏距离、曼哈顿距离、闵可夫斯基距离
    · 在KNeighborsClassifier的参数中：默认p=2，metric='minkowski'
    · p=2时的minkowski距离即为欧氏距离，p=1时的minkowski距离即为曼哈顿距离（绝对值）
    · 故默认的距离计算方式是欧氏距离
- k值对算法结果的影响：
    k值取得过小，容易受到异常点的影响
    k值取得过大，容易受到样本不均衡的影响
- knn算法涉及计算距离，需要事先进行无量纲化处理（标准化）
- API：
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
n_neighbors (default=5): k值
algorithm {'auto', 'ball_tree', 'kd_tree', 'brute'}: auto自动决定最合适的算法
"""
# 案例1：鸢尾花种类预测
# 1. 获取数据
iris = datasets.load_iris()

# 先划分数据集后特征工程，标准化统一使用训练集数据，测试集也使用训练集参数
# 2. 数据集划分
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=10)

# 3. 特征工程标准化
# test集使用transform方法：使用的是上一步train集的参数直接计算
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. KNN预估器流程
estimator = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
estimator.fit(x_train, y_train)

# 5. 模型评估
# 方法一：predict()后直接比对
y_predict = estimator.predict(x_test)
print("y_predict: \n", y_predict)
print("直接比对真实值和预测值: \n", y_test == y_predict)

# 方法二：score()计算准确率
accuracy = estimator.score(x_test, y_test)
print("准确率: \n", accuracy)

