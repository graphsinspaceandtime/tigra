from tigra.evaluation.clbase.classifier_base import Classifier

from typing import Sequence
from sklearn import tree, svm, neighbors, naive_bayes, ensemble, linear_model

class SKLCModel(Classifier):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def predict_proba(self, train_data):
        return self.model.predict_proba(train_data)

    

class SVM_SKLC(SKLCModel):
    def __init__(self, kernel="rbf"):
        super().__init__("SVM", svm.SVC(kernel=kernel))


class RF_SKLC(SKLCModel):
    def __init__(self, numEstimators=5):
        cm = ensemble.RandomForestClassifier(n_estimators=numEstimators)
        super().__init__("RF", cm)


class NB_SKLC(SKLCModel):
    def __init__(self):
        super().__init__("NB", naive_bayes.GaussianNB())


class KNN_SKLC(SKLCModel):
    def __init__(self, numNeighbors=5):
        cm = neighbors.KNeighborsClassifier(n_neighbors=numNeighbors)
        super().__init__("KNN", cm)


class DT_SKLC(SKLCModel):
    def __init__(self):
        super().__init__("DT", tree.DecisionTreeClassifier())


class LR_SKLC(SKLCModel):
    def __init__(self, penalty=None):
        super().__init__("LOGREG", linear_model.LogisticRegression(penalty=penalty))


class AdaBoost_SKLC(SKLCModel):
    def __init__(self, estimator=tree.DecisionTreeClassifier(max_depth=10), n_estimators=100):
        super().__init__(
            "AdaBoost", model=ensemble.AdaBoostClassifier(
                estimator=estimator, n_estimators=n_estimators)
        )


def skl_classifiers_factory() -> Sequence[SKLCModel]:
    return [
        LR_SKLC(),
        DT_SKLC(),
        SVM_SKLC(),
        KNN_SKLC(),
        RF_SKLC(),
        NB_SKLC(),
        AdaBoost_SKLC()
    ]