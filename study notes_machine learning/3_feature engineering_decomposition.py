import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
# 降维：降低特征的个数，得到不相关的主变量
"""
1. Feature Selection
1.1 Filter过滤式
    Variance：过滤掉低方差特征（某一特征在样本中方差低意味着普遍或趋同，不适宜为特征）
    Correlation Coefficient：特征与特征之间r越低越好
1.2 Embedded嵌入式
    Decision Tree
    Regularization
    Deep Learning
2. PCA
"""

""" 1. Filter feature with low variance
sklearn.feature_selection.VarianceThreshold(threshold = 0.0)
Variance.fit_transform(X)
X: (n_samples, n_features)
return: 删除训练集差异低于threshold的特征
默认threshold为0.0，即删除所有样本具有相同值的特征
"""
# 1. 获取数据
data = pd.read_csv("factor_returns.csv")
data = data.iloc[:, 1: -2]
print("data: \n", data)
# 2. 实例化一个转换器类
transfer1 = VarianceThreshold(threshold = 15)
# 3. 调用fit_transform
data_new1 = transfer1.fit_transform(data)
print("data_new低方差特征过滤：\n", data_new1, data_new1.shape)


"""
2. Deal with features of high correlation
皮尔森Pearson相关系数 -1<=r<=+1, 绝对值越大相关性越强
|r|<0.4低度相关, 0.4<=|r|<=0.7显著性相关, 0.7<=|r|<1高度线性相关
使用scipy.stats中pearsonr
传入x,y两array，返回Pearson系数r和p值
如果特征之间相关性很高：
    1）选取其中一个
    2）加权求和
    3）主成分分析
"""
r_p1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
print("pe_radio & pb_ratio：\n", r_p1)
r_p2 = pearsonr(data["revenue"], data["total_expense"])
print("revenue & total_expense: \n", r_p2)

"""
3. PCA：高维压缩至低维，损失少量信息
sklearn.decomposition.PCA(n_components)
n_components:
1) 小数：表示保留百分之多少的信息
2）整数：减少到几个特征
"""
# 1. 创建一个转换器类
transfer2 = PCA(n_components=2)
# 2. 调用fit_transform
data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
data_new2 = transfer2.fit_transform(data)
print("data_new2主成分分析：\n", data_new2)