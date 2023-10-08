from sklearn.naive_bayes import  MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Line 73前：为正常使用fetch的代码
import os
import codecs
import pickle
import tarfile
import requests
from tqdm import tqdm
import sklearn.datasets
from os.path import splitext
from sklearn.datasets import load_files

def _pkl_filepath(*args, **kwargs):
    """Return filename for Python 3 pickles

    args[-1] is expected to be the ".pkl" filename. For compatibility with
    older scikit-learn versions, a suffix is inserted before the extension.

    _pkl_filepath('/path/to/folder', 'filename.pkl') returns
    '/path/to/folder/filename_py3.pkl'

    """
    py3_suffix = kwargs.get("py3_suffix", "_py3")
    basename, ext = splitext(args[-1])
    basename += py3_suffix
    new_args = args[:-1] + (basename + ext,)
    return os.path.join(*new_args)


data_home = sklearn.datasets.get_data_home()

CACHE_NAME = "20news-bydate.pkz"
TRAIN_FOLDER = "20news-bydate-train"
TEST_FOLDER = "20news-bydate-test"

cache_path =  _pkl_filepath(data_home, CACHE_NAME)
twenty_home = os.path.join(data_home, "20news_home")

def download_newsgroups_data():
    file_url = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
    file_size = int(requests.head(file_url).headers["content-length"])

    res = requests.get(file_url, stream=True)
    pbar = tqdm(total=file_size, unit="B", unit_scale=True)

    archive_path = os.path.join(twenty_home, "20news-bydate.tar.gz")

    with open(archive_path, 'wb') as file:
        for chunk in res.iter_content(chunk_size=1024):
            file.write(chunk)
            pbar.update(len(chunk))
        pbar.close()

    tarfile.open(archive_path, "r:gz").extractall(path=twenty_home)
    os.remove(archive_path)

    # Store a zipped pickle
    train_path = os.path.join(twenty_home, TRAIN_FOLDER)
    test_path = os.path.join(twenty_home, TEST_FOLDER)

    cache = dict(train=load_files(train_path, encoding='latin1'),
                 test=load_files(test_path, encoding='latin1'))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

# download_newsgroups_data()


"""
Naive Bayes Model：朴素 + 贝叶斯的概率相关算法：最后分到概率最大的类别中
Naive：assume features are independent from each other
Bayes：P(C|W) = P(W|C)*P(C) / P(W)
W: 联合多个特征
C: 目标值（类别）
应用场景：文本分类，单词作为特征 / 假设单词之间相互独立 make sense

API:
1. 实例化
sklearn.naive_bayes.MultinomialNB(alpha = 1.0)
alpha: 拉普拉斯平滑系数
2. 调用fit()方法
"""

# Classify news with Naive Bayes：
# 1）获取数据
news = fetch_20newsgroups(subset="all")
# print(news)

# 2）划分数据集
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

# 3）特征工程：文本特征抽取-tfidf
transfer = TfidfVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4）朴素贝叶斯算法预估器流程
estimator = MultinomialNB()
estimator.fit(x_train, y_train)

# 5）模型评估
# 方法一：predict()后直接比对
y_predict = estimator.predict(x_test)
print("y_predict: \n", y_predict)
print("直接比对真实值和预测值: \n", y_test == y_predict)

# 方法二：score()计算准确率
accuracy = estimator.score(x_test, y_test)
print("准确率: \n", accuracy)