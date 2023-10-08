# Titanic - Machine Learning from Disaster
# Predict which passengers would survive in Titanic shipwreck

# 1. Import data
import pandas as pd
titanic = pd.read_csv("titanic.csv")

# 2. Extract features and targets
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

# 3. Preprocess data
# 1）Replace null values
x['age'].fillna(x['age'].mean(), inplace=True)

# 2）Transfer features to dict for future use
x = x.to_dict(orient='records')
print(x)

# 4. Divide datasets into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 5. Dict feature extraction
from sklearn.feature_extraction import DictVectorizer
transfer = DictVectorizer(sparse=False)
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
print(x_train)
print(transfer.get_feature_names_out())

# 6. DecisionTree Estimator
from sklearn.tree import DecisionTreeClassifier, export_graphviz
estimator = DecisionTreeClassifier(criterion='entropy')
# Grid Search & Cross Validation for parameter 'max_depth'
from sklearn.model_selection import GridSearchCV
param_dict = {'max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]}
estimator = GridSearchCV(estimator, param_grid=param_dict)
estimator.fit(x_train, y_train)

# 7. Evaluate model
# 1) Direct Comparison
y_predict = estimator.predict(x_test)
print("y_predict: \n", y_predict)
print("direct_comp_result：\n", y_predict == y_test)

# or 2) Score
score = estimator.score(x_test, y_test)
print("accurate_rate：\n", score)

# Visualiza DecisionTree
export_graphviz(estimator.best_estimator_, out_file='titanic_tree.dot', feature_names=transfer.feature_names_)