from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

"""
1. Normalization
1) X' = (x - min) / (max - min)
2) X'' = X' * (mx - mi) + mi
作用于每一列（每一特征），mx mi分别为指定区间
"""

# 1. Get data
data = pd.read_csv("dating.txt")
data = data.loc[:, ["milage", "Liters", "Consumtime"]]
print(data)

# 2. Instantiate a class
# feature_range默认[0,1]
transfer_1 = MinMaxScaler(feature_range=(2, 3))

# 3. fit_transform
# 传入二维列表，[n_samples x n_features]
data_new1 = transfer_1.fit_transform(data)
print("data_new归一化：\n", data_new1)


"""
2. Standardization
X' = (x - mean) / std
标准化比归一化更robust，受异常值影响较小
标准化后，数据均值0，std 1
"""

# 1. StandardScaler（）
transfer_2 = StandardScaler()

# 2. fit_transform
data_new2 = transfer_2.fit_transform(data)
print(data_new2)

