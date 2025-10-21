from abc import abstractmethod

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class Classifier:
    @abstractmethod
    def train(self, train_data, train_labels):
        pass

    @abstractmethod
    def predict(self, test_data):
        pass

    def evaluate(self, data, labels):
        (
            train_data,
            test_data,
            train_labels,
            test_labels,
        ) = train_test_split(data, labels, test_size=0.33)

        self.train(train_data, train_labels)
        predicted_labels = self.predict(test_data)

        return (
            accuracy_score(test_labels, predicted_labels),
            precision_score(
                test_labels, predicted_labels, average="macro", zero_division=0
            ),
            recall_score(
                test_labels, predicted_labels, average="macro", zero_division=0
            ),
            f1_score(test_labels, predicted_labels, average="macro", zero_division=0),
        )
