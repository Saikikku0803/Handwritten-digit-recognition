from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from config import IMAGE_SIZE

def train_random_forest(x_train, y_train, n_estimators=10, criterion='entropy', random_state=3):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state,
        n_jobs=2
    )
    clf.fit(x_train, y_train)
    return clf

def evaluate_model(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    correct = (y_pred == y_test).sum()
    total = len(y_test)
    accuracy = correct / total
    return accuracy, pd.DataFrame({'actual': y_test, 'prediction': y_pred})

def predict_custom_images(clf, x_custom, y_custom):
    y_pred = clf.predict(x_custom)
    correct = (y_pred == y_custom).sum()
    total = len(y_custom)
    accuracy = correct / total
    result_df = pd.DataFrame({'actual': y_custom, 'prediction': y_pred})
    return accuracy, result_df
