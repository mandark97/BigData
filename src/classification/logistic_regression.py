import json
import os

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

MATH_DATASET = "student-alcohol-consumption/student-mat.csv"
POR_DATASET = "student-alcohol-consumption/student-por.csv"
STUDENTS_DATASET = "student-alcohol-consumption/students.csv"

ex = Experiment("logistic_regression")

# if ran from the container the url is "mongodb://mongo:27017"
ex.observers.append(MongoObserver.create(
    url="mongodb://localhost:27017",
    db_name="sacred"
))
ex.add_config("src/classification/config.json")


@ex.config
def model_config():
    clf_params = {
        "classifier__C": np.arange(0.1, 1),
        "classifier__tol": [1e-4, 1e-2, 1e-3, 1e-5],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__n_jobs": [-1],
        "classifier__verbose": [2],
        "classifier__multi_class": ["auto"]
    }


def numeric_preprocessor():
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])


def categorical_preprocessor():
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])


@ex.capture
def preprocessor_transformer(categorical_features=[], numeric_features=[]):
    return ColumnTransformer(transformers=[
        ('num', numeric_preprocessor(), numeric_features),
        ('cat', categorical_preprocessor(), categorical_features)
    ])


@ex.capture
def select_columns(df, categorical_features=[], numeric_features=[], label=[]):
    features = categorical_features + numeric_features

    return df[features], df[label]


def store_dict(data, name):
    with open(f"metrics/{name}.json", "w") as f:
        json.dump(data, f)
    ex.add_artifact(f"metrics/{name}.json")


@ex.automain
def main(dataset, clf_params):
    df = pd.read_csv(dataset)
    X, y = select_columns(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    preprocess = preprocessor_transformer()
    clf = Pipeline(steps=[
        ('preprocessor', preprocess),
        ('classifier', LogisticRegression())
    ])
    grid = GridSearchCV(clf,
                        clf_params,
                        n_jobs=-1,
                        cv=5,
                        verbose=2,
                        return_train_score=True,
                        refit=True)

    grid.fit(X_train, y_train)
    score = grid.score(X_test, y_test)

    print('score=', score)
    ex.log_scalar('score', score)
    ex.log_scalar('best_score', grid.best_score_)

    y_pred = grid.predict(X_test)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs("metrics", exist_ok=True)
    store_dict(class_report, "classification_report")
    store_dict(grid.best_params_, "best_params")

    results = pd.DataFrame(grid.cv_results_)
    results.to_csv("metrics/cv_results.csv")
    ex.add_artifact("metrics/cv_results.csv")

    joblib.dump(grid.best_estimator_, "metrics/model.joblib")
    ex.add_artifact("metrics/model.joblib")
