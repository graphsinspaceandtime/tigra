import numpy as np
import networkx as nx

from tigra.core.dgraph import DynamicAggregateGraph
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr
from pyod.models.auto_encoder import AutoEncoder


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

        iforest = IsolationForest(contamination=0.1, random_state=42)
        iforest.fit(X)
        iforest_score = iforest.decision_function(X)

        aencoder = AutoEncoder()
        aencoder.fit(X)
        aencoder_score = aencoder.decision_scores_

        degree = nx.degree_centrality(lightcopyg)
        betweenes = nx.betweenness_centrality(lightcopyg)
        eigenvector = nx.eigenvector_centrality(lightcopyg)

        centrality_scores = np.array([degree[node] for node in nodes_list])

        corr_lof_cen, p_lof_cen = spearmanr(lof_score, centrality_scores)
        corr_iso_cen, p_iso_cen = spearmanr(iforest_score, centrality_scores)
        corr_aen_cen, p_aen_cen = spearmanr(aencoder_score, centrality_scores)

        betweenes_scores = np.array([betweenes[node] for node in nodes_list])
        corr_lof_btw, p_lof_btw = spearmanr(lof_score, betweenes_scores)
        corr_iso_btw, p_iso_btw = spearmanr(iforest_score, betweenes_scores)
        corr_aen_btw, p_aen_btw = spearmanr(aencoder_score, betweenes_scores)

        eigenvector_scores = np.array([eigenvector[node] for node in nodes_list])
        corr_lof_egn, p_lof_egn = spearmanr(lof_score, eigenvector_scores)
        corr_iso_egn, p_iso_egn = spearmanr(iforest_score, eigenvector_scores)
        corr_aen_egn, p_aen_egn = spearmanr(aencoder_score, eigenvector_scores)

        rez = f"{corr_lof_cen},{p_lof_cen},{corr_iso_cen},{p_iso_cen},{corr_aen_cen},{p_aen_cen},{corr_lof_btw},{p_lof_btw},{corr_iso_btw},{p_iso_btw},{corr_aen_btw},{p_aen_btw},{corr_lof_egn},{p_lof_egn},{corr_iso_egn},{p_iso_egn},{corr_aen_egn},{p_aen_egn}"

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
