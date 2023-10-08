from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba

# A. Datasets
# 1. Get Datasets: load/fetch
# Small-scale：load_*
iris = datasets.load_iris()
# Large-scale：fetch_*
# first parameter: data_home (default: ~/scikit_learn_data/)
# subset: train / test / all
# print(sklearn.datasets.fetch_20newsgroups(data_home=None, subset='train'))

# 2. Return
description = iris["DESCR"]
# print(description)
target_names = iris.target_names
# print(target_names)
# key/value OR .attribute
""" 
load/fetch return：datasets.base.Bunch
data: 特征数据数组，[n_samples * n_features] 2-D numpy.ndarray
target：标签数组，是n——samples的一维numpy.ndarray数组
DESCR：数据描述
feature_names: 特征名
target_names: 标签名"""

# 3. 数据集的划分：训练组+测试组
# sklearn.model_selection.train_test_split(arrays, *options)
"""
1. x数据集特征值
2. y数据集标签值
3. test_size 测试集大小，一般为float，如0.2
4. random_state 随机数种子，相同种子采样结果相同
5. return：训练集特征值，测试集特征值，训练集目标值，测试集目标值
            x_train     x_test      y_train     y_test
"""
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=15)
# print(x_test)


"""B. Feature Engineering
Feature Extraction：将任意数据（文本/图像）转换成数值（特征值化）
sklearn.feature_extraction
1. Dict Feature Extraction（特征离散化）
目标：转换为one-hot编码
使用feature_extraction的DictVectorizer类
首先实例化，再使用DictVectorize.fit_transform(X)方法
X：字典或包含字典的迭代器 / 返回值：sparse矩阵（onehot编码的非零值的位置信息）
"""
# 1. Instantiate class
# 默认sparse=True，表示return稀疏矩阵，改成False，return onehot
transfer_dic = DictVectorizer(sparse=False)
data_dic = [{"city": "Beijing", "temperature": 100},
     {"city": "Shanghai", "temperature": 60},
     {"city": "Shenzhen", "temperature": 30}]
# 2. fit_transform()
data_dic_new = transfer_dic.fit_transform(data_dic)
print("data_dict_new: \n", data_dic_new)
# 3. DictVectorizer.inverse_transform(X): 返回转换之前的(data_dict)数据格式
data_dict_test = transfer_dic.inverse_transform(X=data_dic_new)
print("data_dict_test: \n", data_dict_test)
# 4. Dictvectorizer.get_feature_names_out(): 返回类别名称
print("特征名字：\n", transfer_dic.get_feature_names_out())

"""
2. Text Feature Extraction
使用feature_extraction.text的CountVectorizer类（count特征单词出现的次数）
注意：count以空格为间隔识别单词并计数，如果是中文文本需要手动添加空格分隔词汇
首先实例化，再使用CountVectorizer.fit_transform(X)方法
X：字符或字符迭代器 / 返回值：sparse矩阵
CountVectorizer没有sparse TF的参数设l置，但经过fit_transform得到的矩阵
有方法toarray()可以把sparse矩阵转换成二维数组
"""

data_char = ["Life is short, I like like Python, the best interpreter",
             "Life is long long, I dislike Python"]
# stop_words参数：停用词，把自然语言中用处不大的词汇通过列表传入，使其不成为特征词汇
transfer_char = CountVectorizer(stop_words=["is", "the"])
data_char_new = transfer_char.fit_transform(data_char)
# inverse_transform返回每项的特征词汇列表（没有计数）
data_char_test = transfer_char.inverse_transform(X=data_char_new)
print("data_char_new: \n", data_char_new.toarray())
print("data_char_test: \n", data_char_test)
print("特征名字：\n", transfer_char.get_feature_names_out())

"""
3. Chinese Text Feature Extraction: automatically divide elements
jieba库方法cut()

"""
def cut_words(text):
    return " ".join(list(jieba.cut(text)))

data_charChinese = ["今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
     "我们看到的从很远星系来的光是在几百万年前之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
     "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

data_charChinese_new = []
for sent in data_charChinese:
    data_charChinese_new.append(cut_words(sent))
print("data_charChinese_new: \n", data_charChinese_new)

transfer_charChinese = CountVectorizer(stop_words=["是", "这样", "的"])
data_charChinese_final = transfer_charChinese.fit_transform(data_charChinese_new)
print("data_charChinese_final: \n", data_charChinese_final.toarray())

"""
4. TFidfVectorizer: 衡量一字词在文本中的重要程度（在该文本中出现次数多，但在其他文本中出现次数少，即关键词）
TF - term frequency 词频：词语出现次数 / 文章总词数
Idf - inverse document frequency 逆向文档频率：一个词语普遍重要性的度量
总文件数目除以包含该词语的文件数目，再将商取以10为底的对数
tf * idf 得到 表示词语重要程度
"""
transfer_charChinese_tfidf = TfidfVectorizer(stop_words=["是", "这样", "的"])
data_charChinese_final_tfidf = transfer_charChinese_tfidf.fit_transform(data_charChinese_new)
print("data_charChinese_final_tfidf: \n", data_charChinese_final_tfidf.toarray())
print("data_charChinese_final_tfidf: \n", data_charChinese_final_tfidf.toarray())
print("tfidf_feature_names: \n", transfer_charChinese_tfidf.get_feature_names_out(), transfer_charChinese_tfidf.get_feature_names_out().size)


