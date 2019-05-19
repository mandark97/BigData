import pandas as pd
from dotenv import load_dotenv
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
        "solver": "lbfgs",
        "n_jobs": -1,
        "verbose": 2,
    }


@ex.capture
def preprocessor(categorical_features, numeric_features):
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


@ex.capture
def select_columns(df, categorical_features, numeric_features, label):
    selected_columns = categorical_features + numeric_features

    return df[selected_columns], df[label]


@ex.automain
def main(dataset, clf_params):
    df = pd.read_csv(dataset)
    X, y = select_columns(df)
    # y = LabelBinarizer().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    preprocess = preprocessor()

    clf = Pipeline(steps=[
        ('preprocessor', preprocess),
        ('classifier', LogisticRegression(**clf_params))
    ])
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    print('score=', score)
    ex.log_scalar('score', score)
