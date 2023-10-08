"""
K-means Clustering
1. 随机设置k个特征空间内的点作为初始聚类中心（k-超参数）
2. 对于其他每个点计算到K个中心的距离，未知的点选择最近的聚类中心作为标记类别
3. 重新计算出每个聚类的新中心点（平均值）
4. 如果计算得出的新中心点与原中心点一致，结束操作，否则回到第二步重复过程

K-means API
sklearn.cluster.KMeans(n_clusters=8, n_init='k-means++')
    n_clusters: 开始的聚类中心数量
    n_init: 初始化方法，默认为'k-means++'
    labels_: 返回标记的类型，可以和真实值比较

预估器：实例化后，fit(), predict()

Kmeans evaluation: silhouette score
SCi = (b_i - a_i) / max(b_i, a_i)
每个点i为已聚类数据中的样本，b_i为i到其它族群的所有样本的距离最小值，a_i为i到本身簇的距离平均值
最终计算出所有样本点的轮廓系数平均值
希望：高内聚，低耦合，即外部距离最大化，内部距离最小化
Aim: maximize external distance + minimize internal distance
若b_i>>a_i, 趋近于1，效果好
b_i<<a_i, 趋近于-1，效果差
轮廓系数的值【-1，1】，越趋近于1说明内聚度和分离度都越优秀

轮廓系数API
sklearn.metrics.silhouette_score(X, labels)
    X: 特征值
    labels: 被聚类标记的目标值
"""