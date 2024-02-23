import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def cal_dist(df1, df2):
    return round(np.linalg.norm(df1.values - df2.values, 2), 2)


def rf_acc(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # model = RandomForestRegressor(n_estimators=100, random_state=0)
    model = LogisticRegression(random_state=0, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return round(accuracy_score(y_pred, y_test), 2)