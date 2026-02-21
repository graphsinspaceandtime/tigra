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
        self.target_k = numNeighbors
        cm = neighbors.KNeighborsClassifier(n_neighbors=numNeighbors)
        super().__init__("KNN", cm)

    def fit(self, train_data, train_labels):
        n_samples = len(train_data)
        
        # If we have fewer samples than the requested K, lower K temporarily
        if n_samples < self.target_k:
            self.model.n_neighbors = n_samples
        else:
            self.model.n_neighbors = self.target_k
            
        self.model.fit(train_data, train_labels)
        

class DT_SKLC(SKLCModel):
    def __init__(self):
        super().__init__("DT", tree.DecisionTreeClassifier())


class LR_SKLC(SKLCModel):
    def __init__(self, penalty=None):
        l1_r = 0.5 if penalty == "elasticnet" else None
        
        super().__init__(
            "LOGREG", 
            linear_model.LogisticRegression(
                penalty=penalty, 
                solver='saga',       
                l1_ratio=l1_r,       
                max_iter=10000
            )
        )


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


def rf_tuning_factory() -> Sequence[SKLCModel]:
    return [
        RF_SKLC(numEstimators=1),
        RF_SKLC(numEstimators=2),
        RF_SKLC(numEstimators=3), 
        RF_SKLC(numEstimators=4), 
        RF_SKLC(numEstimators=5), 
        RF_SKLC(numEstimators=10), 
        RF_SKLC(numEstimators=15), 
        RF_SKLC(numEstimators=20), 
        RF_SKLC(numEstimators=25),
        RF_SKLC(numEstimators=50)
    ]


def knn_tuning_factory() -> Sequence[SKLCModel]:
    return [
        KNN_SKLC(numNeighbors=1), 
        KNN_SKLC(numNeighbors=2), 
        KNN_SKLC(numNeighbors=3), 
        KNN_SKLC(numNeighbors=4), 
        KNN_SKLC(numNeighbors=5), 
        KNN_SKLC(numNeighbors=10), 
        KNN_SKLC(numNeighbors=15), 
        KNN_SKLC(numNeighbors=20)
    ]


def svm_tuning_factory() -> Sequence[SKLCModel]:
    return [
        SVM_SKLC(kernel="rbf"), 
        SVM_SKLC(kernel="linear"), 
        SVM_SKLC(kernel="poly"), 
        SVM_SKLC(kernel="sigmoid")
    ]


def lr_tuning_factory() -> Sequence[SKLCModel]:
    return [
        LR_SKLC(penalty=None), 
        LR_SKLC(penalty="l1"), 
        LR_SKLC(penalty="l2"), 
        LR_SKLC(penalty="elasticnet")
    ]