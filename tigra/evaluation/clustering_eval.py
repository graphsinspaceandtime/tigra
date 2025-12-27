"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Cluster based evaluation of dynamic graph embedding methods

author: jelab@dmi.uns.ac.rs
"""

import numpy as np
import networkx as nx
import os

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community import (
    girvan_newman,
    louvain_communities,
    asyn_lpa_communities,
)

from tigra.core.dgraph import DynamicAggregateGraph


class EmbClusterer:
    def __init__(self, graph, embedding, num_com=None):
        self.graph = graph
        self.embedding = embedding
        self.nodes_list = list(self.graph.nodes())
        self.X = np.array([embedding[node] for node in self.nodes_list])
        self.num_coms = num_com

        self.clustering_algorithms = [
            (KMeans(random_state=42, n_init=10, n_clusters=self.num_coms)),
            (AgglomerativeClustering(n_clusters=self.num_coms)),
        ]

        self.results = []

    def detect_clusters(self):
        for alg in self.clustering_algorithms:
            alg_name = alg.__class__.__name__

            labels = alg.fit_predict(self.X)

            lab = set(labels)
            num_labs = len(lab)
            part = []
            for _ in range(num_labs):
                part.append([])

            for i in range(len(self.nodes_list)):
                node = self.nodes_list[i]
                cl = labels[i]
                part[cl].append(node)

            mod = modularity(self.graph, part)
            if labels is not None:
                num_clusters = len(set(labels))
            else:
                num_clusters = "N/A"

            labels_dictionary = {}
            for i in labels:
                labels_dictionary[i] = labels[i]

            self.results.append(
                {
                    "alg": alg_name,
                    "communities": num_clusters,
                    "silhouette_score": self.silhouette_score(self.X, labels),
                    "labels": labels_dictionary,
                    "modularity": mod,
                }
            )

    def silhouette_score(self, X, labels):
        if len(set(labels)) <= 1:
            return -1
        return silhouette_score(X, labels)


class GraphClusterer:
    def __init__(self, graph):
        self.graph = graph
        self.G = self.graph.to_undirected()
        self.node_to_index = {node: idx for idx, node in enumerate(self.G.nodes())}

        self.results = []

    def get_gnc(self, graph):
        communities_generator = girvan_newman(graph)
        communities_list = []
        for com in communities_generator:
            communities_list.append(com)

        return communities_list

    def get_lp(self, graph):
        iter = asyn_lpa_communities(graph, seed=0)
        part = []
        for c in iter:
            part.append(c)

        return part

    def get_gmc(self, graph):
        return np.asarray(greedy_modularity_communities(graph))

    def louvian(self, graph):
        return louvain_communities(graph, seed=0)

    def detect_communities(self):
        algorithms = [
            ("lp", self.get_lp),
            ("Louvain", self.louvian),
            ("GMO", self.get_gmc),
        ]

        for name, method in algorithms:
            communities = method(self.G)
            mod = modularity(self.G, communities)
            labels = self.com_labels(communities)

            self.results.append(
                {
                    "alg": name,
                    "communities": communities,
                    "modularity": mod,
                    "labels": labels,
                }
            )

        return self.results

    def com_labels (self, communities):
        labels = {}
        for community_id, community in enumerate(communities):
            for node in community:
                labels[self.node_to_index[node]] = community_id

        return labels


class ClusteringEval:
    def __init__(self, embedding, graph_sequence, dataset, f):
        # if isinstance(dg, DynamicAggregateGraph):
        #     self.graph_sequence = [dg.get_aggregate_graph()]
        # else:
        #     self.graph_sequence = dg.get_graph_sequence()

        self.graph_sequence = graph_sequence
        self.embedding = embedding
        self.dataset = dataset
        self.path = f

    def eval_snapshot(self, emb, graph):

        lightcopyg = graph.G

        grc = GraphClusterer(lightcopyg)
        grc.detect_communities()

        best_ginfo = max(grc.results, key=lambda ginfo: ginfo["modularity"])
        num_com = len(best_ginfo["communities"])

        embc = EmbClusterer(lightcopyg, emb, num_com=num_com)
        embc.detect_clusters()

        return grc.results, embc.results

    def eval(
        self,
    ):
        N = len(self.graph_sequence)
        algorithms = ["lp", "Louvain", "GMO", "KMeans", "AgglomerativeClustering"]
        labels_by_alg = {alg: [] for alg in algorithms}

        for i in range(N):
            try:
                print(f"snapshot number: {i}")
                grc, embc = self.eval_snapshot(
                    self.embedding[i], self.graph_sequence[i]
                )
                ginfo_dict = {}
                einfo_dict = {}
                for g in grc:
                    ginfo_dict[g["alg"]] = g

                for e in embc:
                    einfo_dict[e["alg"]] = e

                row = [i]
                for alg in algorithms:
                    ginfo = ginfo_dict.get(alg, None)
                    einfo = einfo_dict.get(alg, None)

                    if ginfo:
                        row.extend([ginfo["modularity"], len(ginfo["communities"])])
                        current_labels = ginfo_dict[alg]["labels"]
                        labels_by_alg[alg].append(current_labels)
                        
                    if einfo:
                        row.extend([einfo["modularity"], einfo["communities"]])
                        current_labels = einfo_dict[alg]["labels"]
                        labels_by_alg[alg].append( current_labels)

                    if alg in ["KMeans", "AgglomerativeClustering"]:
                        row.append(einfo["silhouette_score"])

                self.save_results_to_csv(row)
            except Exception as e:
                print(f"An error occurred: {e}")
        print("-------------------\n")
        
        self.nmi_calc(labels_by_alg)

    def save_results_to_csv(self, row):
        with open(self.path, "a", encoding="utf-8") as f:
            output = ", ".join(map(str, row))

            f.write(f"{output}\n")
            print(f"{output}")
            print(f"Results saved to file")
            
    def nmi_calc(self, labels_by_alg):
        path = os.path.join("clustering_eval_stwalk_nmi", f"{self.dataset}.csv")
        with open(path, "a", encoding="utf-8") as f:
            f.write("ALG,SNP1,SNP2,NMI,\n")
            for alg, label_dict_list in labels_by_alg.items():
                for i in range(1, len(label_dict_list)):
                    prev_dict = label_dict_list[i - 1]
                    curr_dict = label_dict_list[i]
                    common_keys = set(prev_dict.keys()) & set(curr_dict.keys())
                    if common_keys:
                        prev_labels = [prev_dict[node] for node in common_keys]
                        curr_labels = [curr_dict[node] for node in common_keys]
                        nmi = normalized_mutual_info_score(prev_labels, curr_labels)
                        f.write(f"{alg},{i-1},{i},{nmi:.4f}\n")
                    else:
                        print(f"Snapshot {i-1} and {i} has no shared nodes. NMI could not be calculated.")
                    print("-------------------")
