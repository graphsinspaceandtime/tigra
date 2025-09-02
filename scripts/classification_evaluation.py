import os
import sys
from contextlib import contextmanager

sys.path.append("../..")

from tigra.embeddings.embedding import FSEmbedding
from tigra.evaluation.classification_eval import eval

from tigra.evaluation.clbase.nn_torch import (
    NeuralNetworkClassifier,
)

from tigra.evaluation.clbase.skl import (
    DT_SKLC,
    SVM_SKLC,
    KNN_SKLC,
    NB_SKLC,
    RF_SKLC,
)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def eval_dataset_classifier(dataset, classifier):
    with suppress_stdout():
        emb = FSEmbedding("../" + dataset)

    (score_accuracy, score_precision, score_recall, score_f1) = eval(emb, classifier)

    print(
        f"{dataset}\t{classifier.__class__.__name__}\t{score_accuracy}\t{score_precision}\t{score_recall}\t{score_f1}"
    )


def eval_all():
    datasets = [
        "ia-hospital-ward-proximity-attr",
        "ia-primary-school-proximity-attr",
        "dpp_labeled",
        "dbac_labeled",
    ]

    classifiers = [
        DT_SKLC(),
        SVM_SKLC(),
        KNN_SKLC(),
        NB_SKLC(),
        RF_SKLC(),
        NeuralNetworkClassifier(),
    ]

    print("Dataset\tClassifier\tAccuracy\tPrecision\tRecall\tF1")

    for dataset in datasets:
        for classifier in classifiers:
            eval_dataset_classifier(dataset, classifier)


if __name__ == "__main__":
    eval_all()
