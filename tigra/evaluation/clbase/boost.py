from tigra.evaluation.clbase.classifier_base import Classifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


class XGB(Classifier):
    def __init__(self, n_estimators=4):
        super().__init__("XGB")
        self.model = XGBClassifier(n_estimators=n_estimators, objective='binary:logistic')

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def predict_proba(self, train_data):
        return self.model.predict_proba(train_data)
    

class CAT(Classifier):
    def __init__(self):
        super().__init__("CAT")
        self.model = CatBoostClassifier(logging_level='Silent')

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def predict_proba(self, train_data):
        return self.model.predict_proba(train_data)
    

class LGBM(Classifier):
    def __init__(self):
        super().__init__("LGBM")
        self.model = LGBMClassifier(verbose=-1)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def predict_proba(self, train_data):
        return self.model.predict_proba(train_data)