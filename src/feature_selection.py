import pandas as pd

MATH_DATASET = "student-alcohol-consumption/student-mat.csv"
POR_DATASET = "student-alcohol-consumption/student-mat.csv"
STUDENTS_DATASET = "student-alcohol-consumption/students.csv"


def select_features(self, df, feature_names=[], label_names=[]):
    if len(set(feature_names) & set(label_names)) > 0:
        raise Exception("Features should not be used as labels")

    X = df[feature_names]
    y = df[label_names]

    return X, y
