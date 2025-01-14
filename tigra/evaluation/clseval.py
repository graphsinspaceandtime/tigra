"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Cluster based evaluation of dynamic graph embedding methods

author: jelab@dmi.uns.ac.rs
"""

import numpy as np

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity, coverage


class EmbClusterer:
    def __init__(self, graph, embedding, num_clusters, community_labels):

        self.graph = graph
        self.embedding = embedding
        self.num_clusters = num_clusters
        self.nodes_list = list(self.graph.nodes())
        self.X = np.array([embedding[node] for node in self.nodes_list])
        self.community_labels = community_labels

        self.clustering_algorithms = [
            KMeans(n_clusters=num_clusters, random_state=42),
            DBSCAN(eps=0.5, min_samples=5),
            AgglomerativeClustering(n_clusters=num_clusters, linkage="complete"),
        ]

        self.results = {}

    def detect_clusters(self):

        for alg in self.clustering_algorithms:
            alg_name = alg.__class__.__name__

            clusters = alg.fit_predict(self.X)

            unique_labels = set(clusters)
            if len(unique_labels) > 1 and -1 not in unique_labels:
                sil_score = silhouette_score(self.X, clusters)
            else:
                sil_score = -1
            nmi = normalized_mutual_info_score(self.community_labels, clusters)
            self.results[alg_name] = {"nmi": nmi, "silhouette": sil_score}

    def print_results(self):
        for alg_name, info in self.results.items():
            print(f"Algorithm: {alg_name}")
            print(f"  Labels: {info['nmi']}")
            print(f"  Silhouette Score: {info['silhouette']:.3f}")
            print()


class GraphClusterer:
    def __init__(self, graph):
        self.graph = graph
        G = self.graph.to_undirected()
        self.communities = np.asarray(greedy_modularity_communities(G))
        self.num_communities = len(self.communities)
        self.q = modularity(G, self.communities)

    def get_num_communities(self):
        return self.num_communities

    def get_modularity(self):
        return self.q

    def get_communities(self):
        return self.communities

    def get_community_labels(self):
        cdict = dict()
        for i in range(len(self.communities)):
            for n in self.communities[i]:
                cdict[n] = i

        clabels = []
        for n in self.graph.nodes:
            clabels.append(cdict[n])

        return clabels


class ClusteringEval:
    def __init__(self, embedding, dg):
        self.embedding = embedding
        self.graph_sequence = dg.get_graph_sequence()

    def eval_snapshot(self, index, emb, graph):

        lightcopyg = graph.G
        # lightcopyg.remove_edges_from(nx.selfloop_edges(lightcopyg))

        grc = GraphClusterer(lightcopyg)

        num_communities = grc.get_num_communities()
        modularity = grc.get_modularity()
        community_labels = grc.get_community_labels()

        s = str(index + 1) + ","
        s += str(num_communities) + ","
        s += str(modularity)

        embc = EmbClusterer(lightcopyg, emb, num_communities, community_labels)
        embc.detect_clusters()
        embc.print_results()

        return s

    def eval(
        self,
    ):
        N = len(self.graph_sequence)
        for i in range(N):
            print(i)
            s = self.eval_snapshot(i, self.embedding[i], self.graph_sequence[i])
            print(s)
        print("-------------------\n")
