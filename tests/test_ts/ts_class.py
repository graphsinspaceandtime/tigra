import sys
import os
import pickle

sys.path.append(os.path.join("..", ".."))
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from tigra.embeddings.embedding import FSEmbedding
import pickle

from tigra.evaluation.classification_eval import eval
from tigra.evaluation.clbase.skl import skl_classifiers_factory

emb_folders = [
    "Plane_10_5_1.0_graph_dn2v",
    "Plane_20_5_1.0_graph_dn2v",
    "Plane_50_1_1.0_graph_dn2v",
    "Plane_50_5_0.1_graph_dn2v",
    "Plane_50_5_0.05_graph_dn2v",
    "Plane_50_10_1.0_graph_dn2v",
    "Plane_10_5_1.0_graph_stwalk",
    "Plane_20_5_1.0_graph_stwalk",
    "Plane_50_1_1.0_graph_stwalk",
    "Plane_50_5_0.1_graph_stwalk",
    "Plane_50_5_0.05_graph_stwalk",
    "Plane_50_10_1.0_graph_stwalk",
    "Plane_10_5_1.0_graph_deep_hub",
    "Plane_20_5_1.0_graph_deep_hub",
    "Plane_50_1_1.0_graph_deep_hub",
    "Plane_50_5_0.1_graph_deep_hub",
    "Plane_50_5_0.05_graph_deep_hub",
    "Plane_50_10_1.0_graph_deep_hub",
    "Plane_10_5_1.0_graph_deep_lid",
    "Plane_50_5_0.1_graph_deep_lid",
    "Plane_50_5_0.05_graph_deep_lid",
    "Plane_50_10_1.0_graph_deep_lid",
]

header = "model\taccuracy\tprecision\trecall\tf1\n"

if __name__ == "__main__":

    with open("rezultati.txt", "a", encoding="utf-8") as f:

        for folder_name in emb_folders:

            if not os.path.isdir(folder_name):
                continue

            emb = FSEmbedding(folder_name)
            f.write(f"{folder_name}\n")
            f.flush()
            os.fsync(f.fileno())

            for classifier in skl_classifiers_factory():
                (score_accuracy, score_precision, score_recall, score_f1) = eval(
                    emb, classifier
                )

                line = f"{classifier.__class__.__name__}\t{score_accuracy}\t{score_precision}\t{score_recall}\t{score_f1}\n"
                f.write(header)
                f.write(f"{line}\n\n")
                f.flush()
                os.fsync(f.fileno())
