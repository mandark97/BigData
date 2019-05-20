import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

MATH_DATASET = "student-alcohol-consumption/student-mat.csv"
POR_DATASET = "student-alcohol-consumption/student-mat.csv"
STUDENTS_DATASET = "student-alcohol-consumption/students.csv"

load_dotenv()
ex = Experiment()

ex.observers.append(MongoObserver.create(
    url="mongodb://mongo_user:mongo_password@localhost:27017/?authSource=admin",
    db_name="sacred"
))


@ex.config
def preproces_config():
    dataset = POR_DATASET
    categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                            'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
                            'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                            'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health']
    numeric_features = ['age', 'absences', 'G1', 'G2', 'G3']
    label = 'Dalc'


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
    with open(f"{name}.json", "w") as f:
        json.dump(data, f)
    ex.add_artifact(f"{name}.json")


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

    store_dict(class_report, "classification_report")
    store_dict(grid.best_params_, "best_params")

    results = pd.DataFrame(grid.cv_results_)
    results.to_csv("cv_results.csv")
    ex.add_artifact("cv_results.csv")
