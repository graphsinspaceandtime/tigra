import sys

from numpy.ma.extras import average

from tests.mars.neural_network import NeuralNetworkClassifier
from tests.mars.utils import suppress_stdout
from tigra.evaluation.classification.skl_classifiers import (
    DecisionTree,
    SVM,
    KNN,
    NaiveBayes,
    RandomForest,
)

sys.path.append("../..")

from tigra.embeddings.embedding import FSEmbedding


def eval(embedding, classifier):
    scores_accuracy = []
    scores_precision = []
    scores_recall = []

    for e in embedding:
        for i in range(10):
            node_ids = e.get_node_ids()
            node_vectors = [e[node_id] for node_id in node_ids]
            labels = [e.get_label(node_id) for node_id in node_ids]

            (
                score_accuracy,
                score_precision,
                score_recall,
                score_f1,
            ) = classifier.evaluate(node_vectors, labels)

            scores_accuracy.append(score_accuracy)
            scores_precision.append(score_precision)
            scores_recall.append(score_recall)

    score_accuracy = average(scores_accuracy)
    score_precision = average(scores_precision)
    score_recall = average(scores_recall)
    score_f1 = 2 / (1 / score_precision + 1 / score_recall)

    return (score_accuracy, score_precision, score_recall, score_f1)


def do():
    datasets = [
        "ia-hospital-ward-proximity-attr",
        "ia-primary-school-proximity-attr",
        "dpp_labeled",
        "dbac_labeled",
    ]

    classifiers = [
        DecisionTree(),
        SVM(),
        KNN(),
        NaiveBayes(),
        RandomForest(),
        NeuralNetworkClassifier(),
    ]

    print("Dataset\tClassifier\tAccuracy\tPrecision\tRecall\tF1")

    for dataset in datasets:
        with suppress_stdout():
            embedding = FSEmbedding("../" + dataset)

        for classifier in classifiers:
            (score_accuracy, score_precision, score_recall, score_f1) = eval(
                embedding, classifier
            )
            print(
                f"{dataset}\t{classifier.__class__.__name__}\t{score_accuracy}\t{score_precision}\t{score_recall}\t{score_f1}"
            )


if __name__ == "__main__":
    do()
