
from tigra.evaluation.clbase.classifier_base import Classifier

from tabpfn import TabPFNClassifier


class TPFN(Classifier):
    def __init__(self):
        super().__init__("TabPFN")
        self.model = TabPFNClassifier()

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def predict_proba(self, train_data):
        return self.model.predict_proba(train_data)