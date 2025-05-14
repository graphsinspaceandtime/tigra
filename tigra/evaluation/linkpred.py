from abc import ABC, abstractmethod
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicGraph, Graph
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import random
from networkx.algorithms.link_prediction import jaccard_coefficient, adamic_adar_index
import json

# supervised learning imports
from tigra.evaluation.classification.classifier import Classifier

from typing import Sequence, Literal


class NamedOperation(ABC):
    name: str

    def __init__(self, name):
        self.name = name


class LinkPrediction(ABC):
    result: dict
    _prediction_type: Literal["unsupervised", "supervised"]

    @abstractmethod
    def predict(self, embedding: Embedding, graph: DynamicGraph, snapshot: int) -> dict:
        pass

    def _sample_links(
        self, embedding: SingletonEmbedding, graph_t: Graph, graph_tp1: Graph
    ):
        positive_links = []
        negative_links = []

        for u in embedding.embedding.keys():
            for v in embedding.embedding.keys():
                if u >= v:
                    continue

                if graph_tp1.has_edge(u, v) and not graph_t.has_edge(u, v):
                    positive_links.append((u, v))
                elif not graph_tp1.has_edge(u, v) and not graph_t.has_edge(u, v):
                    negative_links.append((u, v))

        return positive_links, negative_links

    def _negative_sample_links(
        self, embedding: SingletonEmbedding, graph_t: Graph, graph_tp1: Graph
    ):
        positive_links, negative_links = self._sample_links(
            embedding, graph_t, graph_tp1
        )
        if len(positive_links) < len(negative_links):
            negative_links = random.sample(negative_links, len(positive_links))
        elif len(positive_links) > len(negative_links):
            positive_links = random.sample(positive_links, len(negative_links))

        return (
            positive_links,
            negative_links,
        )

    @abstractmethod
    def print_result(self):
        pass

    @property
    @abstractmethod
    def _operation(self) -> NamedOperation:
        pass


class Distance(NamedOperation):
    positive_distances: list
    negative_distances: list

    _distance_type: str

    def __init__(self, name: str):
        super().__init__(name)

    @abstractmethod
    def measure(
        self,
        positive_links: list,
        negative_links: list,
        *args,
        **kwargs,
    ):
        pass

    @property
    def scores(self):
        return [*self.positive_distances, *self.negative_distances]

    @property
    def labels(self):
        return [1] * len(self.positive_distances) + [0] * len(self.negative_distances)


class EmbeddingDistance(Distance):
    name: str
    positive_distances: list
    negative_distances: list

    def __init__(self, name: str):
        self.name = name
        self._distance_type = "embedding"

    @abstractmethod
    def _measure(self, u_hat, v_hat) -> float:
        pass

    def measure(self, positive_links, negative_links, embedding: SingletonEmbedding):
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


class DotProduct(EmbeddingDistance):
    def __init__(self):
        super().__init__("DotProduct")

    def _measure(self, u_hat, v_hat):
        return np.dot(u_hat, v_hat)


class CosineSimilarity(EmbeddingDistance):
    def __init__(self):
        super().__init__("CosineSimilarity")

    def _measure(self, u_hat, v_hat):
        return np.dot(u_hat, v_hat) / (np.linalg.norm(u_hat) * np.linalg.norm(v_hat))


class EuclideanDistance(EmbeddingDistance):
    def __init__(self):
        super().__init__("EuclideanDistance")

    def _measure(self, u_hat, v_hat):
        return float(-np.linalg.norm(np.array(u_hat) - np.array(v_hat)))


class L1_Distance(EmbeddingDistance):
    def __init__(self):
        super().__init__("L1_Distance")

    def _measure(self, u_hat, v_hat):
        return np.sum(np.abs(np.array(u_hat) - np.array(v_hat)))


class StructuralDistance(Distance):
    name: str
    positive_distances: list
    negative_distances: list

    def __init__(self, name: str):
        self.name = name
        self._distance_type = "structural"

    @abstractmethod
    def _measure(
        self, positive_links: list, negative_links: list, graph: Graph
    ) -> tuple:
        pass

    def measure(self, positive_links, negative_links, graph: Graph):
        positive_distances, negative_distances = self._measure(
            positive_links, negative_links, graph
        )
        self.positive_distances = [distance[2] for distance in positive_distances]
        self.negative_distances = [distance[2] for distance in negative_distances]

    @property
    def scores(self):
        return [*self.positive_distances, *self.negative_distances]

    @property
    def labels(self):
        return [1] * len(self.positive_distances) + [0] * len(self.negative_distances)


class JaccardCoefficient(StructuralDistance):
    def __init__(self):
        super().__init__("JaccardCoefficient")

    def _measure(self, positive_links, negative_links, graph):
        return (
            jaccard_coefficient(graph.G, positive_links),
            jaccard_coefficient(graph.G, negative_links),
        )


class AdamicAdarIndex(StructuralDistance):
    def __init__(self):
        super().__init__("AdamicAdarIndex")

    def _measure(self, positive_links, negative_links, graph):
        return (
            adamic_adar_index(graph.G, positive_links),
            adamic_adar_index(graph.G, negative_links),
        )


def distance_measure_factory() -> Sequence[Distance]:
    return [
        DotProduct(),
        CosineSimilarity(),
        EuclideanDistance(),
        L1_Distance(),
        JaccardCoefficient(),
        AdamicAdarIndex(),
    ]


class UnsupervisedLinkPrediction(LinkPrediction):
    distance: Distance

    def __init__(self, distance: Distance):
        super().__init__()
        self.distance = distance
        self._prediction_type = "unsupervised"

    def predict(self, embedding, graph, snapshot) -> dict:
        singleton_embedding = embedding.embeddings[snapshot]
        if singleton_embedding == None:
            raise Exception("Embedding is None")

        graph_sequence = graph.get_graph_sequence()
        graph_t = graph_sequence[snapshot]
        graph_tp1 = graph_sequence[snapshot + 1]

        positive_links, negative_links = self._sample_links(
            singleton_embedding, graph_t, graph_tp1
        )

        if self.distance._distance_type == "embedding":
            self.distance.measure(
                positive_links, negative_links, embedding=singleton_embedding
            )
        elif self.distance._distance_type == "structural":
            self.distance.measure(positive_links, negative_links, graph=graph_t)

        self.result = {
            "roc_auc": roc_auc_score(self.distance.labels, self.distance.scores),
            "average_precision": average_precision_score(
                self.distance.labels, self.distance.scores
            ),
        }

        return self.result

    def print_result(self):
        roc_auc = self.result["roc_auc"]
        average_precision = self.result["average_precision"]
        print(f"ROC AUC: {roc_auc}")
        print(f"Average precision: {average_precision}")

    @property
    def _operation(self):
        return self.distance


class CombinePairOperation(NamedOperation):
    positive_embeddings: list
    negative_embeddings: list

    def __init__(self, name: str):
        super().__init__(name)

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
    combiner: CombinePairOperation
    classifier: Classifier

    def __init__(self, combiner: CombinePairOperation, classifier: Classifier):
        self.combiner = combiner
        self.classifier = classifier
        self._prediction_type = "supervised"

    def predict(self, embedding, graph, snapshot):
        singleton_embedding_km1 = embedding.embeddings[snapshot - 1]
        singleton_embedding_k = embedding.embeddings[snapshot]

        if singleton_embedding_km1 == None:
            raise Exception("Embedding is None")
        if singleton_embedding_k == None:
            raise Exception("Embedding is None")

        graph_sequence = graph.get_graph_sequence()
        graph_km1 = graph_sequence[snapshot - 1]
        graph_k = graph_sequence[snapshot]
        graph_kp1 = graph_sequence[snapshot + 1]

        positive_links_train, negative_links_train = self._negative_sample_links(
            singleton_embedding_km1, graph_km1, graph_k
        )

        self.combiner.combine(
            positive_links_train, negative_links_train, singleton_embedding_km1
        )

        X_train, y_train = np.array(self.combiner.embeddings), np.array(
            self.combiner.labels
        )

        positive_links_test, negative_links_test = self._negative_sample_links(
            singleton_embedding_k, graph_k, graph_kp1
        )

        self.combiner.combine(
            positive_links_test, negative_links_test, singleton_embedding_k
        )

        X_test, y_test = np.array(self.combiner.embeddings), np.array(
            self.combiner.labels
        )

        y_pred = self.classifier.fit_predict(X_train, y_train, X_test)
        (
            res_f1,
            res_accuracy,
            res_precision,
            res_recall,
            res_pos_prec,
            res_pos_rec,
            res_neg_prec,
            res_neg_rec,
        ) = self.classifier.score(y_test, y_pred, print_results=False)
        self.result = {
            "f1": res_f1,
            "accuracy": res_accuracy,
            "precision": res_precision,
            "recall": res_recall,
            "pos_prec": res_pos_prec,
            "pos_rec": res_pos_rec,
            "neg_prec": res_neg_prec,
            "neg_rec": res_neg_rec,
        }

        return self.result

    def print_result(self):
        f1 = self.result["f1"]
        accuracy = self.result["accuracy"]
        precision = self.result["precision"]
        recall = self.result["recall"]
        pos_prec = self.result["pos_prec"]
        pos_rec = self.result["pos_rec"]
        neg_prec = self.result["neg_prec"]
        neg_rec = self.result["neg_rec"]

        print(f"F1 score: {f1}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"+Precision: {pos_prec}")
        print(f"+Recall: {pos_rec}")
        print(f"-Precision: {neg_prec}")
        print(f"-Recall: {neg_rec}")

    @property
    def _operation(self):
        return self.combiner


class Predicter:
    linkpred: LinkPrediction
    embedding: Embedding
    dg: DynamicGraph

    results = []

    def __init__(
        self, linkpred: LinkPrediction, embedding: Embedding, dg: DynamicGraph
    ):
        self.linkpred = linkpred
        self.embedding = embedding
        self.dg = dg

    def predict(self):
        graph_sequence = dg.get_graph_sequence()
        num_graphs = len(graph_sequence)
        num_embeddings = self.embedding.num_embeddings()

        self.results = []

        if self.linkpred._prediction_type == "unsupervised":
            if num_graphs < 2 or num_embeddings < 1:
                raise Exception("Insufficient number of graphs or embeddings")

            for i in range(num_embeddings - 1):
                result = self.linkpred.predict(self.embedding, self.dg, i)
                self.results.append(result)
        else:
            if num_graphs < 3 or num_embeddings < 2:
                raise Exception("Insufficient number of graphs or embeddings")

            for i in range(1, num_embeddings - 1):
                result = self.linkpred.predict(self.embedding, self.dg, i)
                self.results.append(result)

    def save_report(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prediction_type": self.linkpred._prediction_type,
                    "operation": self.linkpred._operation.name,
                    "results": self.results,
                },
                f,
            )

    def print_csv_columns(self):
        for result in self.results:
            print(result["f1"])
        return
        columns = [score for score in self.results[0].keys()]
        for column in columns:
            print(column)
            for result in self.results:
                print(result[column])
            print()
            print()
            print()
            print()


if __name__ == "__main__":
    from tigra.embeddings.embedding import FSEmbedding
    from tigra.data.loaders import RadoslawEmail

    # embed = FSEmbedding(
    #     folder_name="../../scripts/dusan/make_embed/embs_ae/radoslaw_email"
    # )
    embed = FSEmbedding(folder_name="../../tests/radoslaw-dynn2v-25")
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
        unsup_linkpred = UnsupervisedLinkPrediction(distance_measure)
        predicter = Predicter(unsup_linkpred, embed, dg)
        predicter.predict()
        predicter.save_report(
            f"../../scripts/dusan/results/unsupervised_{distance_measure.name}.json"
        )
        print()

    # for classifier in skl_classifiers_factory():
    #     print("==================")
    #     print("Classifier: " + classifier.name)
    #     print("==================")
    #     for combine_pair_operation in combine_pair_operation_factory():
    #         sup_linkpred = SupervisedLinkPrediction(combine_pair_operation, classifier)
    #         print("Combine pair operation: " + combine_pair_operation.name)
    #         predicter = Predicter(sup_linkpred, embed, dg)
    #         predicter.predict()
    #         # predicter.print_csv_columns()
    #         predicter.save_report(f"../../scripts/dusan/results/supervised_{classifier.name}_{combine_pair_operation.name}.json")
    #         print()
