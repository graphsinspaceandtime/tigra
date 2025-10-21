from abc import ABC, abstractmethod
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold


class Classifier(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name


    @abstractmethod
    def fit(self, train_data, train_labels):
        pass


    @abstractmethod
    def predict(self, test_data):
        pass


    def fit_predict(self, train_data, train_labels, test_data):
        self.fit(train_data, train_labels)
        return self.predict(test_data)


    """
    training-test split evaluation
    """
    def evaluate_ttsplit(self, data, labels, test_size=0.33, average="macro", zero_division=0):
        (
            train_data,
            test_data,
            train_labels,
            test_labels,
        ) = train_test_split(data, labels, test_size=test_size)

        self.fit(train_data, train_labels)
        predicted_labels = self.predict(test_data)

        return (
            accuracy_score(test_labels, predicted_labels),
            precision_score(
                test_labels, predicted_labels, average=average, zero_division=zero_division
            ),
            recall_score(
                test_labels, predicted_labels, average=average, zero_division=zero_division
            ),
            f1_score(test_labels, predicted_labels, average=average, zero_division=zero_division),
        )
    

    """
    evaluation of binary classifiers for given training and test sets
    """
    def evaluate_binarycl(self, trainX, trainY, testX, testY):
        self.fit(trainX, trainY)
        predictions = self.predict(testX)
        
        acc = accuracy_score(testY, predictions)
        prec = precision_score(testY, predictions, average="macro", zero_division=0)
        rec = recall_score(testY, predictions, average="macro", zero_division=0)
        f1 = f1_score(testY, predictions, average="macro", zero_division=0)
        
        pos_prec = precision_score(
            testY, predictions, average="binary", pos_label=1, zero_division=0
        )
        pos_rec = recall_score(
            testY, predictions, average="binary", pos_label=1, zero_division=0
        )
        neg_prec = precision_score(
            testY, predictions, average="binary", pos_label=0, zero_division=0
        )
        neg_rec = recall_score(
            testY, predictions, average="binary", pos_label=0, zero_division=0
        )

        return (f1, acc, prec, rec, pos_prec, pos_rec, neg_prec, neg_rec)
    

    """
    k-fold cross-validation 
    """
    def evaluate_kfold_cv(self, data, labels, n_splits=10, random_state=42, average="macro", zero_division=0):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        acc, prec, rec, f1 = [], [], [], []
        data = np.array(data)
        labels = np.array(labels)

        # iterate through folds
        for train_index, test_index in kf.split(data, labels):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # Initialize and train the model for the current fold
            self.fit(X_train, y_train)

            # Make predictions and evaluate
            y_pred = self.predict(X_test)
            
            # compute evaluation scores
            accs = accuracy_score(y_test, y_pred)
            acc.append(accs)

            precs = precision_score(y_test, y_pred, average=average, zero_division=zero_division)
            prec.append(precs)

            recs = recall_score(y_test, y_pred, average=average, zero_division=zero_division)
            rec.append(recs)

            f1s = f1_score(y_test, y_pred, average=average, zero_division=zero_division)
            f1.append(f1s)


        # return averages
        return (
            np.mean(acc),
            np.mean(prec),
            np.mean(rec),
            np.mean(f1)
        )


