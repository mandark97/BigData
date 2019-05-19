import pandas as pd
from dotenv import load_dotenv
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
import json
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
        "solver": "lbfgs",
        "n_jobs": -1,
        "verbose": 2,
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
def preprocessor_transformer(categorical_features, numeric_features):
    return ColumnTransformer(transformers=[
        ('num', numeric_preprocessor(), numeric_features),
        ('cat', categorical_preprocessor(), categorical_features)
    ])


@ex.capture
def select_columns(df, categorical_features, numeric_features, label):
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
        ('classifier', LogisticRegression(**clf_params))
    ])

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('score=', score)
    ex.log_scalar('score', score)

    y_pred = clf.predict(X_test)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    with open("classification_report.json", "w") as f:
        json.dump(class_report, f)
    ex.add_artifact("classification_report.json")
