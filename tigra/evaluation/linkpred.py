from abc import ABC, abstractmethod
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicGraph, Graph
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import random

# supervised learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tigra.evaluation.classification.classifier import Classifier
from tigra.evaluation.classification.skl import skl_classifiers_factory

from typing import Sequence

# TODO: Implement link prediction for multiple embeddings


class LinkPrediction(ABC):
    embedding: Embedding
    dg: DynamicGraph

    def __init__(self, embedding: Embedding, dg: DynamicGraph):
        self.embedding = embedding
        self.dg = dg

    def _sample_links(self, embedding: SingletonEmbedding, graph: Graph):
        positive_links = []
        negative_links = []

        for u in embedding.embedding.keys():
            for v in embedding.embedding.keys():
                if u >= v:
                    continue

                if graph.has_edge(u, v):
                    positive_links.append((u, v))
                else:
                    negative_links.append((u, v))

        return positive_links, negative_links

    def _stratified_sample_links(self, embedding: SingletonEmbedding, graph: Graph):
        positive_links, negative_links = self._sample_links(embedding, graph)
        if len(positive_links) < len(negative_links):
            negative_links = random.sample(negative_links, len(positive_links))
        elif len(positive_links) > len(negative_links):
            positive_links = random.sample(positive_links, len(negative_links))

        return (
            positive_links,
            negative_links,
        )


class Distance(ABC):
    name: str
    positive_distances: list
    negative_distances: list

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def _measure(self, u_hat, v_hat) -> float:
        pass

    def measure(
        self, positive_links: list, negative_links: list, embedding: SingletonEmbedding
    ):
        self.positive_distances = []
        self.negative_distances = []

        for u, v in positive_links:
            u_hat = embedding.embedding[u]
            v_hat = embedding.embedding[v]
            self.positive_distances.append(self._measure(u_hat, v_hat))

        for u, v in negative_links:
            u_hat = embedding.embedding[u]
            v_hat = embedding.embedding[v]
            self.negative_distances.append(self._measure(u_hat, v_hat))

    @property
    def scores(self):
        return [*self.positive_distances, *self.negative_distances]

    @property
    def labels(self):
        return [1] * len(self.positive_distances) + [0] * len(self.negative_distances)


class DotProduct(Distance):
    def __init__(self):
        super().__init__("DotProduct")

    def _measure(self, u_hat, v_hat):
        return np.dot(u_hat, v_hat)


class CosineSimilarity(Distance):
    def __init__(self):
        super().__init__("CosineSimilarity")

    def _measure(self, u_hat, v_hat):
        return np.dot(u_hat, v_hat) / (np.linalg.norm(u_hat) * np.linalg.norm(v_hat))


class EuclideanDistance(Distance):
    def __init__(self):
        super().__init__("EuclideanDistance")

    def _measure(self, u_hat, v_hat):
        return float(-np.linalg.norm(np.array(u_hat) - np.array(v_hat)))


class HadamardProduct(Distance):
    def __init__(self):
        super().__init__("hadamardProduct")

    def _measure(self, u_hat, v_hat):
        return float(np.sum(np.array(u_hat) * np.array(v_hat)))


class L1_Distance(Distance):
    def __init__(self):
        super().__init__("L1_Distance")

    def _measure(self, u_hat, v_hat):
        return np.sum(np.abs(np.array(u_hat) - np.array(v_hat)))


def distance_measure_factory() -> Sequence[Distance]:
    return [
        DotProduct(),
        CosineSimilarity(),
        EuclideanDistance(),
        HadamardProduct(),
        L1_Distance(),
    ]


class UnsupervisedLinkPrediction(LinkPrediction):
    def __init__(self, embedding, dg):
        super().__init__(embedding, dg)

    def predict(self, distance: Distance):
        embedding = self.embedding.embeddings[-1]
        if embedding == None:
            raise Exception("Embedding is None")

        graph = self.dg.get_graph_sequence()[-self.embedding.num_embeddings()]

        positive_links, negative_links = self._stratified_sample_links(embedding, graph)

        distance.measure(positive_links, negative_links, embedding)

        roc_auc = roc_auc_score(distance.labels, distance.scores)
        ap = average_precision_score(distance.labels, distance.scores)

        print(f"ROC AUC: {roc_auc}")
        print(f"Average precision: {ap}")


class CombinePairOperation(ABC):
    name: str
    positive_embeddings: list
    negative_embeddings: list

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def _combine(self, u_hat, v_hat) -> np.ndarray:
        pass

    def combine(
        self, positive_links: list, negative_links: list, embedding: SingletonEmbedding
    ):
        positive_embeddings = []
        negative_embeddings = []

        for u, v in positive_links:
            u_hat = embedding.embedding[u]
            v_hat = embedding.embedding[v]
            positive_embeddings.append(self._combine(u_hat, v_hat))

        for u, v in negative_links:
            u_hat = embedding.embedding[u]
            v_hat = embedding.embedding[v]
            negative_embeddings.append(self._combine(u_hat, v_hat))

        self.positive_embeddings = positive_embeddings
        self.negative_embeddings = negative_embeddings

    @property
    def embeddings(self):
        return [*self.positive_embeddings, *self.negative_embeddings]

    @property
    def labels(self):
        return [1] * len(self.positive_embeddings) + [0] * len(self.negative_embeddings)


class ConcatinationCombiner(CombinePairOperation):
    def __init__(self):
        super().__init__("ConcatinationCombiner")

    def _combine(self, u_hat, v_hat):
        return u_hat + v_hat


class AbsoluteDifferenceCombiner(CombinePairOperation):
    def __init__(self):
        super().__init__("AbsoluteDifferenceCombiner")

    def _combine(self, u_hat, v_hat):
        return np.abs(np.array(u_hat) - np.array(v_hat))


class HadamardCombiner(CombinePairOperation):
    def __init__(self):
        super().__init__("HadamardCombiner")

    def _combine(self, u_hat, v_hat):
        return np.array(u_hat) * np.array(v_hat)


def combine_pair_operation_factory() -> Sequence[CombinePairOperation]:
    return [
        ConcatinationCombiner(),
        AbsoluteDifferenceCombiner(),
        HadamardCombiner(),
    ]


class SupervisedLinkPrediction(LinkPrediction):
    def __init__(self, embedding, dg):
        super().__init__(embedding, dg)

    def predict(self, combiner: CombinePairOperation, classifier: Classifier):
        embedding = self.embedding.embeddings[-1]
        if embedding == None:
            raise Exception("Embedding is None")

        graph = self.dg.get_graph_sequence()[-self.embedding.num_embeddings()]

        positive_links, negative_links = self._stratified_sample_links(embedding, graph)

        combiner.combine(positive_links, negative_links, embedding)

        X, y = combiner.embeddings, combiner.labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        y_pred = classifier.fit_predict(X_train, y_train, X_test)
        classifier.score(y_test, y_pred)


if __name__ == "__main__":
    from tigra.embeddings.embedding import FSEmbedding
    from tigra.data.loaders import RadoslawEmail

    embed = FSEmbedding(
        folder_name="../../scripts/dusan/make_embed/embs_ae/radoslaw_email"
    )
    radoslaw = RadoslawEmail("../../data")

    es = radoslaw.get_event_sequence(shrink=True)

    dg = DynamicGraph()
    dg.init_from_event_sequence(
        event_sequence=es,
        time_resolution=radoslaw.resolution,
        directed=False,
        cumulative=radoslaw.cumulative,
    )

    print(f"Number of embeddings: {embed.num_embeddings()}")
    print(f"Number of graphs: {len(dg.get_graph_sequence())}")
    print()

    for distance_measure in distance_measure_factory():
        print("Distance measure: " + distance_measure.name)
        unsup_linkpred = UnsupervisedLinkPrediction(embed, dg)
        unsup_linkpred.predict(distance_measure)
        print()

    sup_linkpred = SupervisedLinkPrediction(embed, dg)
    for classifier in skl_classifiers_factory():
        print("==================")
        print("Classifier: " + classifier.name)
        print("==================")
        for combine_pair_operation in combine_pair_operation_factory():
            print("Combine pair operation: " + combine_pair_operation.name)
            sup_linkpred.predict(combine_pair_operation, classifier)
            print()
