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

    # def _sample_stratified_links(
    #     self, embedding: SingletonEmbedding, graph: Graph
    # ) -> Tuple[Sequence[Tuple[int, int]], Sequence[Tuple[int, int]]]:
    #     num_edges = graph.num_edges()
    #     max_num_edges = round((graph.num_nodes() * (graph.num_nodes() - 1)) / 2)

    #     def make_negative_links() -> Sequence[Tuple[int, int]]:
    #         links: Sequence[Tuple[int, int]] = []

    #         for u in graph.G.nodes:
    #             for v in graph.G.nodes:
    #                 if u >= v:
    #                     continue
    #                 if not graph.has_edge(u, v):
    #                     links.append((u, v))

    #         return links

    #     positive_links = list(graph.G.edges)
    #     negative_links = make_negative_links()

    #     if num_edges < max_num_edges / 2:
    #         negative_links = sample(negative_links, num_edges)
    #     elif num_edges > max_num_edges / 2:
    #         positive_links = sample(list(graph.G.edges), num_edges)

    #     return positive_links, negative_links

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

    def score(self, y_test: np.ndarray, y_pred: np.ndarray, print_results=True):
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

        if print_results:
            print(f"F1 score: {f1}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"+Precision: {pos_prec}")
            print(f"+Recall: {pos_rec}")
            print(f"-Precision: {neg_prec}")
            print(f"-Recall: {neg_rec}")
            # print("ACC,PREC,REC,POS-PREC,POS-REC,NEG-PREC,NEG-REC")
            # print(
            #     f"{accuracy},{precision},{recall},{pos_prec},{pos_rec},{neg_prec},{neg_rec}"
            # )

        return f1, accuracy, precision, recall, pos_prec, pos_rec, neg_prec, neg_rec

    # compatibility functions for svc
    def train(self, predictors, target):
        self.predictors = predictors
        self.target = target
        self.fit(predictors, target)

    def evaluate(self, predictors, target):
        predictions = self.predict(predictors)
        return self.score(target, predictions, print_results=False)

    def trainingScores(self):
        return self.evaluate(self.predictors, self.target)

    def trainingScoresStr(self):
        ev = self.evaluate(self.predictors, self.target)
        return "ACC=" + str(ev[0]) + ", P=" + str(ev[1]) + ", R=" + str(ev[2])
