import json
import os

import pandas as pd
from sklearn.externals import joblib
from yellowbrick.classifier import (ClassificationReport, ClassPredictionError,
                                    ConfusionMatrix)


class ResultStorage(object):
    def __init__(self, experiment, model):
        self.ex = experiment
        self.model = model
        os.makedirs("metrics", exist_ok=True)

    def store_experiment_data(self, X_test, y_test):
        class_report = ClassificationReport(self.model)
        score = class_report.score(X_test, y_test)
        class_report.poof(
            'metrics/classification_report.png', clear_figure=True)
        self.ex.add_artifact('metrics/classification_report.png')

        confustion_matrix = ConfusionMatrix(self.model)
        confustion_matrix.score(X_test, y_test)
        confustion_matrix.poof(
            'metrics/confusion_matrix.png', clear_figure=True)
        self.ex.add_artifact('metrics/confusion_matrix.png')

        cpd = ClassPredictionError(self.model)
        cpd.score(X_test, y_test)
        cpd.poof('metrics/class_prediction_error.png', clear_figure=True)
        self.ex.add_artifact('metrics/class_prediction_error.png')

        print('score=', score)
        self.ex.log_scalar('score', score)
        self.ex.log_scalar('training_score', self.model.best_score_)

        with open(f"metrics/best_params.json", "w") as f:
            json.dump(self.model.best_params_, f)
        self.ex.add_artifact(f"metrics/best_params.json")

        results = pd.DataFrame(self.model.cv_results_)
        results.to_csv("metrics/cv_results.csv")
        self.ex.add_artifact("metrics/cv_results.csv")

        joblib.dump(self.model.best_estimator_, "metrics/model.joblib")
        self.ex.add_artifact("metrics/model.joblib")
