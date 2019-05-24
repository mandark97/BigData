import json
import os

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

from utils import ResultStorage

MATH_DATASET = "../student-alcohol-consumption/student-mat.csv"
POR_DATASET = "../student-alcohol-consumption/student-por.csv"
STUDENTS_DATASET = "../student-alcohol-consumption/students.csv"

ex = Experiment("knn")

# if ran from the container the url is "mongodb://mongo:27017"
ex.observers.append(MongoObserver.create(
    url="mongodb://localhost:27017",
    db_name="sacred"
))
ex.add_config("configs/dataset_config.json")


@ex.config
def model_config():
    if os.path.isfile("configs/knn.json"):
        ex.add_config("configs/knn.json")
    else:
        clf_params = {
            "classifier__n_neighbors": list(np.arange(4, 20, 2)),
            "classifier__leaf_size": list(np.arange(5, 50, 5)),
            "classifier__p": [2, 3],
            "classifier__n_jobs": [-1],
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


@ex.automain
def main(dataset, clf_params):
    df = pd.read_csv(dataset)
    X, y = select_columns(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    preprocess = preprocessor_transformer()
    clf = Pipeline(steps=[
        ('preprocessor', preprocess),
        ('classifier', KNeighborsClassifier())
    ])
    grid = GridSearchCV(clf,
                        clf_params,
                        n_jobs=-1,
                        cv=5,
                        verbose=2,
                        return_train_score=True,
                        refit=True)

    grid.fit(X_train, y_train)

    result_storage = ResultStorage(ex, grid)
    result_storage.store_experiment_data(X_test, y_test)
