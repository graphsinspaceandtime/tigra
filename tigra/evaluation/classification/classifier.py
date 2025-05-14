from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Classifier(ABC):
    name: str

    # compativility fileds for svc
    predictors: np.ndarray
    target: np.ndarray

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        pass

    def score(self, y_test: np.ndarray, y_pred: np.ndarray):
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        pos_prec = precision_score(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0
        )
        pos_rec = recall_score(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0
        )
        neg_prec = precision_score(
            y_test, y_pred, average="binary", pos_label=0, zero_division=0
        )
        neg_rec = recall_score(
            y_test, y_pred, average="binary", pos_label=0, zero_division=0
        )

        return f1, accuracy, precision, recall, pos_prec, pos_rec, neg_prec, neg_rec

    # compatibility functions for svc
    def train(self, predictors, target):
        self.predictors = predictors
        self.target = target
        self.fit(predictors, target)

    def evaluate(self, predictors, target):
        predictions = self.predict(predictors)
        return self.score(target, predictions)

    def trainingScores(self):
        return self.evaluate(self.predictors, self.target)

    def trainingScoresStr(self):
        ev = self.evaluate(self.predictors, self.target)
        return "ACC=" + str(ev[0]) + ", P=" + str(ev[1]) + ", R=" + str(ev[2])
