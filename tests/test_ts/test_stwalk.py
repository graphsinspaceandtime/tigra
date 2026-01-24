import sys
import os
import time

sys.path.append(os.path.join("..", ".."))
sys.path.append(os.path.abspath("."))

sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph, Graph
from tigra.embeddings.embedding import FSEmbedding
from tigra.embeddings.sota.stwalk import STWalk


from tigra.core.greval import GraphReconstructionEval
import pickle

from tigra.evaluation.classification_eval import eval
from tigra.evaluation.classification.neural_network_classifier import (
    NeuralNetworkClassifier,
)
from tigra.evaluation.classification.skl_classifiers import (
    DecisionTree,
    SVM,
    KNN,
    NaiveBayes,
    RandomForest,
)

# from tigra.evaluation import lid_eval

def load_tigra_dgraph(fname):
    with open(fname, "rb") as f:
        tigra_graphs = []
        graphs = pickle.load(f)
        for i, graph in enumerate(graphs):
            tigra_graphs.append(Graph(f"snapshot {i}", nx_graph=graph, labeled=True))
        dyn = DynamicGraph(list_of_graphs = tigra_graphs)
        dyn.info()
        
        return dyn

if __name__ == "__main__":
    dyn = load_tigra_dgraph("Plane_50_10_1.0_graph.pkl")
    dyn.info()
    ts = len(dyn.get_nx_graph_sequence())
    st = STWalk(
    number_restart=40,
    walk_length=10,
    dimension=5, 
    time_step=ts,
    time_window_size=1,
    workers=4,
    vocab_window_size=5,
    )
    st.make_embedding(dyn)
    folder_name = "Plane_50_10_1.0_graph_stwalk"
    st.save_embedding(folder_name)
    
    org_graphs = dyn.get_graph_sequence()
    for i in range(len(org_graphs)):
        print("SNAPSHOT", i)
        emb = st.embeddings[i]
        graph = org_graphs[i]

        gre = GraphReconstructionEval(graph, emb)

        print("Precision", gre.get_avg_precision())
        print("Recall", gre.get_avg_recall())
        print("F1", gre.getF1())

        fn = str(i) + ".labels"
        path = os.path.join(folder_name, fn)
        graph.export_labels(path)
        
    emb = FSEmbedding(folder_name)
    print("#embeddings", len(emb.embeddings))
    for e in emb:
        nodes = e.get_node_ids()
        node_0 = nodes[0]
        emb_0 = e[node_0]
        label_0 = e.get_label(node_0)
        print(node_0, "label", label_0, "emb dimension", len(emb_0))
        print(emb_0)
        print()
    
    classifiers = [
        DecisionTree(),
        SVM(),
        KNN(),
        NaiveBayes(),
        RandomForest(),
        NeuralNetworkClassifier(),
    ]
    
    for classifier in classifiers:
        (score_accuracy, score_precision, score_recall, score_f1) = eval(emb, classifier)
        print(
        f"{classifier.__class__.__name__}\t{score_accuracy}\t{score_precision}\t{score_recall}\t{score_f1}"
        )

