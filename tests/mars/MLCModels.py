import numpy as np

from sklearn import tree, svm, neighbors, naive_bayes, ensemble
from sklearn.metrics import precision_score, recall_score, accuracy_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class SKLCModel:
    def __init__(self, modelName, classifierMethod):
        self.modelName = modelName
        self.classifierMethod = classifierMethod

    def train(self, predictors, target, classes=[]):
        self.predictors = predictors
        self.target = target
        self.classifierMethod.train(predictors, target)

    def evaluate(self, predictors, target):
        predictions = self.classifierMethod.predict(predictors)
        acc = accuracy_score(target, predictions)
        prec = precision_score(target, predictions, average="macro", zero_division=0)
        rec = recall_score(target, predictions, average="macro", zero_division=0)
        return acc, prec, rec

    def trainingScores(self):
        return self.evaluate(self.predictors, self.target)

    def trainingScoresStr(self):
        ev = self.evaluate(self.predictors, self.target)
        return "ACC=" + str(ev[0]) + ", P=" + str(ev[1]) + ", R=" + str(ev[2])

    def predict(self, predictors):
        return self.classifierMethod.predict(predictors)


class SVM_SKLC(SKLCModel):
    def __init__(self):
        super().__init__("SVM", svm.SVC())


class RF_SKLC(SKLCModel):
    def __init__(self, numEstimators=10):
        cm = ensemble.RandomForestClassifier(n_estimators=numEstimators)
        super().__init__("RF", cm)


class NB_SKLC(SKLCModel):
    def __init__(self):
        super().__init__("NB", naive_bayes.GaussianNB())


class KNN_SKLC(SKLCModel):
    def __init__(self, numNeighbors=10):
        cm = neighbors.KNeighborsClassifier(n_neighbors=numNeighbors)
        super().__init__("KNN", cm)


class DT_SKLC(SKLCModel):
    def __init__(self):
        super().__init__("DT", tree.DecisionTreeClassifier())


class TFNNClassifier:
    def __init__(self, hiddenLayersStructure, epochs=10, batchSize=30):
        self.modelName = "TFNN"
        self.epochs = epochs
        self.batchSize = batchSize
        self.hiddenLayersStructure = hiddenLayersStructure

    def initModelStructure(self, predictors, classes):
        self.numPredictors = predictors.shape[1]

        self.targetDict = {}
        self.targetInvDict = {}

        i = 0
        for t in classes:
            self.targetDict[t] = i
            self.targetInvDict[i] = t
            i += 1

        self.numClasses = len(classes)

        self.model = Sequential()
        self.model.add(
            Dense(
                self.hiddenLayersStructure[0],
                activation="relu",
                input_shape=(self.numPredictors,),
            )
        )
        for i in range(1, len(self.hiddenLayersStructure)):
            numNodes = self.hiddenLayersStructure[i]
            self.model.add(Dense(numNodes, activation="relu"))
        self.model.add(Dense(self.numClasses, activation="softmax"))
        self.model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])

    def toOneHot(self, target):
        labid = self.targetDict[target]
        enc = np.zeros(self.numClasses)
        enc[labid] = 1
        return enc

    def prepareData(self, predictors, target):
        normTrainData = self.ss.transform(predictors)
        encodedPredictors = np.array([self.toOneHot(l) for l in target])
        return normTrainData, encodedPredictors

    def train(self, predictors, target, classes):
        self.initModelStructure(predictors, classes)
        self.ss = StandardScaler().fit(predictors)
        normPredictors, normTarget = self.prepareData(predictors, target)
        self.model.fit(
            normPredictors,
            normTarget,
            epochs=self.epochs,
            batch_size=self.batchSize,
            verbose=0,
        )

    def evaluate(self, predictors, target):
        normPredictors, _ = self.prepareData(predictors, target)
        predictions = []
        preds = self.model.predict(normPredictors)
        for p in preds:
            maxIndex = 0
            for i in range(1, len(p)):
                if p[i] > p[maxIndex]:
                    maxIndex = i
            predictions.append(self.targetInvDict[maxIndex])

        acc = accuracy_score(target, predictions)
        prec = precision_score(target, predictions, average="macro", zero_division=0)
        rec = recall_score(target, predictions, average="macro", zero_division=0)
        return acc, prec, rec

    def predict(self, predictors):
        if predictors.ndim == 1:
            predictors = np.expand_dims(predictors, axis=0)
        preds = self.model.predict(self.ss.transform(predictors))
        maxIndices = np.argmax(
            preds, axis=1
        )  # Assume preds.shape = [Batchsize, Features]
        predictions = [self.targetInvDict[maxIndex] for maxIndex in maxIndices]
        return predictions


# if __name__ == "__main__":
#     from core.Datasets import loadDefaultDataset
#     data = loadDefaultDataset()
#
#     predictors, target = data.getData("Depression")
#     tfc = TFNNClassifier([40, 20, 20])
#     tfc.train(predictors, target)
#     print(tfc.evaluate(predictors, target))
#
#     """
#     for i in data.getTargetFields():
#         print(i)
#         predictors, target = data.getData(i)
#
#         model = SVM_SKLC()
#         model.train(predictors, target)
#         print("SVM", model.trainingScoresStr())
#
#         model = RF_SKLC()
#         model.train(predictors, target)
#         print("RF", model.trainingScoresStr())
#
#         model = NB_SKLC()
#         model.train(predictors, target)
#         print("NB", model.trainingScoresStr())
#
#         model = KNN_SKLC()
#         model.train(predictors, target)
#         print("KNN", model.trainingScoresStr())
#
#         model = DT_SKLC()
#         model.train(predictors, target)
#         print("DT", model.trainingScoresStr())
#
#         print("\n")
#     """
