from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import datasets
from sklearn.model_selection import train_test_split
"""
1. Decision Tree：if-else分支结构 层层判断 最终确定类别
2. 决策高效与否的关键：特征的先后顺序
3. 引入信息论基础：
1）信息
   Shannon ：消除随机不确定性
2）信息的衡量：information entropy (the uncertainty among information)
4. Information Gain：
    g(D|A) = H(D) - 条件熵H(D|A)
    A：specific feature
    D: classification
    在给定特征A的情况下的信息熵相较于总体信息熵的减少，
    即已知A带来的信息不确定程度的降低，
    即A带来的信息增益
5. 决策树划分依据
    1） ID3: 信息增益 最大的准则（越大越作为靠前的决策）
    2） C4.5: 信息增益比 最大的准则
    3） CART：分类树 - 基尼系数最小的准则, gini系数和信息熵的衡量标准类似，但计算方式不同
6. API：
sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=None)
criterion: 默认是gini系数，也可以选择信息增益的熵 'entropy'
max_depth: 树的深度大小，树可能会分很细的类去特别好得拟合训练集，设定深度大小可以避免过拟合
    ——树根和叶子之间的最大长度，如果样本量小/特征少可以不设置
random_state: 随机数种子
"""

# Case：鸢尾花分类
# 1. 获取数据集
iris = datasets.load_iris()

# 2. 划分数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

# 3. 决策树预估器
estimator = DecisionTreeClassifier(criterion='entropy', random_state=17)
estimator.fit(x_train, y_train)

# 4. 模型评估
# 1）直接比较predict和真实值
y_predict = estimator.predict(x_test)
print("y_predict: \n", y_predict)
print("直接比对的结果：\n", y_test == y_predict)

# 2）使用score函数计算准确率
result = estimator.score(x_test, y_test)
print("准确率: \n", result)

"""
Decision Tree Visualization
sklearn.tree.export_graphviz()导出DOT格式
tree.export_graphviz(estimator, out_file='tree.dot', feature_names = ['','']
out_file: 导出文件名
feature_names: 特征名

将导出的文件复制 通过网站显示结构：http://webgraphviz.com/
"""
export_graphviz(estimator, out_file='iris_tree.dot', feature_names=iris.feature_names)



