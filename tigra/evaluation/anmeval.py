import numpy as np
import networkx as nx

from tigra.core.dgraph import DynamicAggregateGraph
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# from pyod.models.auto_encoder import AutoEncoder


class AnomaliesEval:
    def __init__(self, embedding, dg, dataset, f):
        if isinstance(dg, DynamicAggregateGraph):
            self.graph_sequence = [dg.get_aggregate_graph()]
        else:
            if len(embedding.embeddings) < len(dg.get_graph_sequence()):
                n = len(dg.get_graph_sequence())
                k = len(embedding.embeddings)
                self.graph_sequence = dg.get_graph_sequence()[n - k : n]
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

        degree = nx.degree_centrality(lightcopyg)
        betweenes = nx.betweenness_centrality(lightcopyg)
        eigenvector = nx.eigenvector_centrality(lightcopyg)

        centrality_scores = np.array([degree[node] for node in nodes_list])
        betweenes_scores = np.array([betweenes[node] for node in nodes_list])
        eigenvector_scores = np.array([eigenvector[node] for node in nodes_list])

        K = max(1, int(0.1 * len(lof_score)))
        top_lof_idx = np.argsort(lof_score)[:K]
        top_lof_scores = lof_score[top_lof_idx]
        corr_top10_lof_cen, _ = spearmanr(top_lof_scores, centrality_scores[top_lof_idx])
        corr_top10_lof_btw, _ = spearmanr(top_lof_scores, betweenes_scores[top_lof_idx])
        corr_top10_lof_egn, _ = spearmanr(top_lof_scores, eigenvector_scores[top_lof_idx])

        top_iso_idx    = np.argsort(iforest_score)[:K]
        top_iso_scores = iforest_score[top_iso_idx]
        corr_top_iso_cen, _ = spearmanr(top_iso_scores, centrality_scores[top_iso_idx])
        corr_top_iso_btw, _ = spearmanr(top_iso_scores, betweenes_scores[top_iso_idx])
        corr_top_iso_egn, _ = spearmanr(top_iso_scores, eigenvector_scores[top_iso_idx])

        auc_lof_cen, auc_lof_btw, auc_lof_egn = self.compute_auc(
            lof_score, centrality_scores, betweenes_scores, eigenvector_scores
        )
        auc_iso_cen, auc_iso_btw, auc_iso_egn = self.compute_auc(
            iforest_score, centrality_scores, betweenes_scores, eigenvector_scores
        )

        spearman_rez = (
            f"{corr_top10_lof_cen},{corr_top_iso_cen},"
            f"{corr_top10_lof_btw},{corr_top_iso_btw},"
            f"{corr_top10_lof_egn},{corr_top_iso_egn}"
        )
        auc_rez = (
            f"{auc_lof_cen},{auc_iso_cen},"
            f"{auc_lof_btw},{auc_iso_btw},"
            f"{auc_lof_egn},{auc_iso_egn}"
        )

        lof_outlier_set = set(np.array(nodes_list)[top_lof_idx])
        iso_outlier_set = set(np.array(nodes_list)[top_iso_idx])

        return spearman_rez, auc_rez, lof_outlier_set, iso_outlier_set

    def peripheral_validation(self, lof_score, iforest_score, centrality_scores, snapshot_idx):
        n = len(centrality_scores)
        threshold = np.percentile(centrality_scores, 10)  # donji decil = periferni
        peripheral_mask = centrality_scores <= threshold
        rest_mask = ~peripheral_mask

        if peripheral_mask.sum() == 0 or rest_mask.sum() == 0:
            print(f"  snapshot {snapshot_idx}: not enough nodes for validation")
            return

        lof_peripheral = lof_score[peripheral_mask].mean()
        lof_rest       = lof_score[rest_mask].mean()
        if_peripheral  = iforest_score[peripheral_mask].mean()
        if_rest        = iforest_score[rest_mask].mean()

        lof_confirmed = lof_peripheral < lof_rest
        if_confirmed  = if_peripheral < if_rest

        print(f"  snapshot {snapshot_idx} | "
              f"LOF peripheral={lof_peripheral:.4f} rest={lof_rest:.4f} "
              f"-> {'CONFIRMED' if lof_confirmed else 'NOT confirmed'} | "
              f"IF peripheral={if_peripheral:.4f} rest={if_rest:.4f} "
              f"-> {'CONFIRMED' if if_confirmed else 'NOT confirmed'}")

        return peripheral_mask

    def compute_auc(self, outlier_scores, centrality_scores, betweenes_scores, eigenvector_scores):
        K = max(1, int(0.1 * len(outlier_scores)))
        labels = np.zeros(len(outlier_scores), dtype=int)
        labels[np.argsort(outlier_scores)[:K]] = 1
        if len(np.unique(labels)) < 2:
            return float("nan"), float("nan"), float("nan")
        auc_cen = roc_auc_score(labels, centrality_scores)
        auc_btw = roc_auc_score(labels, betweenes_scores)
        auc_egn = roc_auc_score(labels, eigenvector_scores)
        return auc_cen, auc_btw, auc_egn

    def eval(self):
        N = len(self.graph_sequence)
        prev_lof_set = None
        prev_iso_set = None

        print(f"peripheral validation: {self.dataset}")
        
        for i in range(N):
            try:
                print(f"snapshot number: {i}")
                jaccard_lof = float("nan")
                jaccard_iso = float("nan")
                    
                sp, auc, lof_set, iso_set = self.eval_snapshot(
                    i, self.embedding[i], self.graph_sequence[i]
                )

                if prev_lof_set is not None:
                    jaccard_lof = len(lof_set & prev_lof_set) / len(lof_set | prev_lof_set)
                    jaccard_iso = len(iso_set & prev_iso_set) / len(iso_set | prev_iso_set)
                                   
                    
                self.save_results_to_csv(i, sp, auc, jaccard_lof, jaccard_iso)


                prev_lof_set = lof_set
                prev_iso_set = iso_set

            except Exception as e:
                print(f"An error occurred: {e}")


    def save_results_to_csv(self, snapshot_num, sp, auc, jaccard_lof, jaccard_iso):
        with open(self.path, "a", encoding="utf-8") as f:
            row = [snapshot_num, sp, auc, jaccard_lof, jaccard_iso]
            output = ", ".join(map(str, row))
            f.write(f"{output}\n")
            print(f"{output}")
            print(f"Results saved to file")
    
    def eval_peripheral_validation(self):
        print(f"dataset: {self.dataset}")
        for i in range(len(self.graph_sequence)):
            try:
                print(f"snapshot number: {i}")
                emb   = self.embedding[i]
                graph = self.graph_sequence[i]
                nodes_list = sorted(graph.G.nodes())
                X = np.array([emb[node] for node in nodes_list])

                lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
                lof.fit(X)
                lof_score = lof.decision_function(X)

                iforest = IsolationForest(contamination=0.1, random_state=42)
                iforest.fit(X)
                iforest_score = iforest.decision_function(X)

                degree = nx.degree_centrality(graph.G)
                centrality_scores = np.array([degree[node] for node in nodes_list])

                self.calculate(lof_score, iforest_score, centrality_scores, i)
            except Exception as e:
                print(f"An error occurred: {e}")
                
    def calculate(self, lof_score, iforest_score, centrality_scores, snapshot_idx):
        threshold       = np.percentile(centrality_scores, 10)
        is_peripheral  = centrality_scores <= threshold
        is_not_peripheral = not is_peripheral

        lof_peripheral = lof_score[is_peripheral].mean()
        lof_rest       = lof_score[is_not_peripheral].mean()
        if_peripheral  = iforest_score[is_peripheral].mean()
        if_rest        = iforest_score[is_not_peripheral].mean()
        
        lof_peripheral = False
        if_peripheral = False
        
        if lof_peripheral < lof_rest:
            lof_peripheral = True
            
        if if_peripheral < if_rest:
            if_peripheral = True

        print(f"snapshot: {snapshot_idx}")
        print(f"LOF peripheral={lof_peripheral:.4f} rest={lof_rest:.4f} - {lof_peripheral}")
        print(f"IF peripheral={if_peripheral:.4f} rest={if_rest:.4f} - {if_peripheral}")