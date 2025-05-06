"""
scikit-learn based classifiers
"""

from sklearn import tree, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from classifier import Classifier


class SKLClassifier(Classifier):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def train(self, train_data, train_labels):
        self._model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self._model.predict(test_data)




class DecisionTree(SKLClassifier):
    def __init__(self):
        super().__init__(tree.DecisionTreeClassifier())

class SVM(SKLClassifier):
    def __init__(self):
        super().__init__(SVC())

class KNN(SKLClassifier):
    def __init__(self):
        super().__init__(KNeighborsClassifier())
        
class NaiveBayes(SKLClassifier):
    def __init__(self):
        super().__init__(GaussianNB())

class RandomForest(SKLClassifier):
    def __init__(self):
        super().__init__(ensemble.RandomForestClassifier())
