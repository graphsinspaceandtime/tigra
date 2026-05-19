import sys
import os
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tigra.core.dgraph import DynamicGraph
from data.bank import Databank
from tigra.core.loaders import load_event_sequence


from tigra.embeddings.ae.autoencoders import DistanceDynAE


def generate_edge_features(emb_obj, G_next):
    X = []
    y = []

    nodes = list(G_next.nodes())
    valid_nodes = set(emb_obj.get_node_ids())

    for u, v in G_next.edges():
        if u in valid_nodes and v in valid_nodes:
            emb_u = np.array(emb_obj.embedding[u])
            emb_v = np.array(emb_obj.embedding[v])

            feature = emb_u * emb_v

            X.append(feature)
            y.append(1)

    num_pos = len(X)
    neg_count = 0

    while neg_count < num_pos:
        u = np.random.choice(nodes)
        v = np.random.choice(nodes)

        if (
            u != v
            and not G_next.has_edge(u, v)
            and u in valid_nodes
            and v in valid_nodes
        ):
            emb_u = np.array(emb_obj.embedding[u])
            emb_v = np.array(emb_obj.embedding[v])

            feature = emb_u * emb_v
            X.append(feature)
            y.append(0)
            neg_count += 1

    return np.array(X), np.array(y)


if __name__ == "__main__":
    DATASET = "radoslaw-email"
    TIME_RES = "month"
    DIM = 50

    print(f"--- Loading {DATASET} ---")
    try:
        databank = Databank()
        es = databank.get_data(DATASET)
    except:
        es, _ = load_event_sequence(DATASET)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=TIME_RES, directed=False)
    graph_sequence = dg.get_graph_sequence()

    print(f"\nGenerating Embeddings using DistanceDynAE (Dim={DIM})...")
    ae = DistanceDynAE(dim=DIM, hidden_dim=DIM * 2, epochs=300, lr=0.01)
    ae.make_embedding(dg)

    print("STARTING TEMPORAL LINK PREDICTION")

    print(f"{'Time Step':<15} | {'ROC-AUC':<10} | {'F1 Score':<10}")
    print("-" * 40)

    auc_scores = []
    f1_scores = []

    for t in range(len(graph_sequence) - 1):
        emb_t = ae.embeddings[t]
        G_next = graph_sequence[t + 1].G

        if emb_t is None or len(G_next.edges()) == 0:
            continue

        X, y = generate_edge_features(emb_t, G_next)

        if len(np.unique(y)) < 2:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        auc_scores.append(auc)
        f1_scores.append(f1)

        print(f"t={t} -> t={t+1:<5} | {auc:.4f}     | {f1:.4f}")

    print(f"Mean ROC-AUC : {np.mean(auc_scores):.4f}")
    print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")
