from tigra.evaluation.linkpred_classification.classifier import Classifier
from sklearn import tree, svm, neighbors, ensemble, naive_bayes
from typing import Sequence


class SKLModel(Classifier):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def fit_predict(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test)


class DecisionTree(SKLModel):
    def __init__(self):
        super().__init__("DecisionTree", model=tree.DecisionTreeClassifier())


class SVM(SKLModel):
    def __init__(self):
        super().__init__("SVM", model=svm.SVC())


class KNN(SKLModel):
    def __init__(self, n_neighbors=5):
        super().__init__("KNN", model=neighbors.KNeighborsClassifier(n_neighbors))


class RandomForest(SKLModel):
    def __init__(self, n_estimators=5):
        super().__init__(
            "RandomForest", model=ensemble.RandomForestClassifier(n_estimators)
        )


class NaiveBayes(SKLModel):
    def __init__(self):
        super().__init__("NaiveBayes", model=naive_bayes.GaussianNB())


def skl_classifiers_factory() -> Sequence[SKLModel]:
    return [
        DecisionTree(),
        SVM(),
        KNN(),
        RandomForest(),
        NaiveBayes(),
    ]
