from sklearn.externals import joblib
import pandas as pd

clf = joblib.load("model.joblib")

feature_importances = clf.steps[1][1].feature_importances_

# get feature_names from one hot encoder
cat_f = clf.steps[0][1].transformers_[1][1].steps[1][1].get_feature_names()
num_f = ["age", "absences", "G1", "G2", "G3"]  # maybe get from config

pd.DataFrame(feature_importances, index=(num_f + cat_f),
             columns=["importantce"]).sort_values("importance", ascending=False)
