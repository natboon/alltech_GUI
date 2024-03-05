import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class Voting : 
    # Load the dataset from a CSV file
    file_path = 'apple_quality.csv'
    data = pd.read_csv(file_path)

    # Extract features and target variable
    X = data[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']]
    y = data['Quality'].astype('category').cat.codes  # Convert labels to numeric values

    # Base models
    logistic_model = LogisticRegression()
    naive_bayes_model = GaussianNB()
    svm_model = SVC(probability=True)  # probability=True for AdaBoost compatibility
    decision_tree_model = DecisionTreeClassifier()
    random_forest_model = RandomForestClassifier(n_estimators=50)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    adaboost_model = AdaBoostClassifier()

    # Fit models before making predictions
    logistic_model.fit(X, y)
    naive_bayes_model.fit(X, y)
    svm_model.fit(X, y)
    decision_tree_model.fit(X, y)
    random_forest_model.fit(X, y)
    mlp_model.fit(X, y)
    adaboost_model.fit(X, y)

    # Bagging
    bagging_clf = BaggingClassifier(logistic_model, n_estimators=20)
    bagging_clf.fit(X, y)

    # Boosting
    boosting_clf = AdaBoostClassifier(logistic_model, n_estimators=30)
    boosting_clf.fit(X, y)

    # Voting Classifier
    voting_clf = VotingClassifier(estimators=[
        ('bagging', bagging_clf),
        ('boosting', boosting_clf),
        ('random_forest', random_forest_model),
        ('logistic', logistic_model),
        ('naive_bayes', naive_bayes_model),
        ('svm', svm_model),
        ('decision_tree', decision_tree_model),
        ('mlp', mlp_model),
        ('adaboost', adaboost_model)
    ], voting='hard')  # 'hard' for majority voting, 'soft' for weighted voting
    voting_clf.fit(X, y)