import numpy as np
import networkx as nx

from tigra.core.dgraph import DynamicAggregateGraph
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import pearsonr


class AnomaliesEval:
    def __init__(self, embedding, dg, dataset, f):
        if isinstance(dg, DynamicAggregateGraph):
            self.graph_sequence = [dg.get_aggregate_graph()]
        else:
            self.graph_sequence = dg.get_graph_sequence()
        self.embedding = embedding
        self.dataset = dataset
        self.path = f

    def eval_snapshot(self, index, emb, graph):
        lightcopyg = graph.G

        nodes_list = sorted(lightcopyg.nodes())
        X = np.array([emb[node] for node in nodes_list])
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        lof.fit(X)
        lof_score = lof.decision_function(X)

        centrality = nx.degree_centrality(lightcopyg)
        centrality_scores = np.array([centrality[node] for node in nodes_list])

        corr_lof, p_lof = pearsonr(lof_score, centrality_scores)

        rez = f"{corr_lof}, {p_lof}"
        return rez

    def eval(
        self,
    ):
        N = len(self.graph_sequence)
        for i in range(N):
            try:
                print(f"snapshot number: {i}")
                rez = self.eval_snapshot(i, self.embedding[i], self.graph_sequence[i])
                self.save_results_to_csv(i, rez)
            except Exception as e:
                print(f"An error occurred: {e}")
        print("-------------------\n")

    def save_results_to_csv(self, snapshot_num, rez):
        with open(self.path, "a", encoding="utf-8") as f:
            row = [snapshot_num, rez]
            output = ", ".join(map(str, row))

            f.write(f"{output}\n")
            print(f"{output}")
            print(f"Results saved to file")
