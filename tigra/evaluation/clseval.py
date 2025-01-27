"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Cluster based evaluation of dynamic graph embedding methods

author: jelab@dmi.uns.ac.rs
"""

import numpy as np

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity, coverage
from networkx.algorithms.community import girvan_newman, louvain_communities


class EmbClusterer:
    def __init__(self, graph, embedding):

        self.graph = graph
        self.embedding = embedding
        self.nodes_list = list(self.graph.nodes())
        self.X = np.array([embedding[node] for node in self.nodes_list])

        self.grid_parms = {
            KMeans: {"n_clusters": [2, 3, 4, 5, 10]},
            DBSCAN: {"eps": [0.1, 0.2, 0.3, 0.4, 0.5], "min_samples": [3, 5, 7, 10]},
            AgglomerativeClustering: {
                "n_clusters": [2, 3, 4, 5, 10],
                "linkage": ["ward", "complete", "average"],
            },
        }

        self.clustering_algorithms = [
            (KMeans(random_state=42), self.grid_parms[KMeans]),
            (DBSCAN(), self.grid_parms[DBSCAN]),
            (
                AgglomerativeClustering(),
                self.grid_parms[AgglomerativeClustering],
            ),
        ]

        self.results = {}

    def detect_clusters(self):

        for alg, grid_param in self.clustering_algorithms:
            alg_name = alg.__class__.__name__

            grid_search = GridSearchCV(
                estimator=alg, param_grid=grid_param, scoring=self.score_fn, cv=3
            )
            grid_search.fit(self.X)

            self.results[alg_name] = {
                "params": grid_search.best_params_,
                "score": grid_search.best_score_,
            }

    def score_fn(self, estimator, X):
        return silhouette_score(X, estimator.labels_)

    def print_results(self):
        results = ""
        for alg_name, info in self.results.items():
            s = f"Algorithm: {alg_name}"
            s += f"  Params: {info['params']}"
            s += f"  Silhouette Score: {info['score']:.3f} \n"

            print(s)

            results += s

        return results


class GraphClusterer:
    def __init__(self, graph):
        self.graph = graph
        self.G = self.graph.to_undirected()

        self.results = {}

    def detect_communities(self):
        algorithms = [
            ("Girvan-Newman", girvan_newman),
            ("Louvain", louvain_communities),
            ("GMO", greedy_modularity_communities),
        ]

        for name, method in algorithms:
            communities = method(self.G)

        return self.results


class ClusteringEval:
    def __init__(self, embedding, dg, dataset, f):
        self.embedding = embedding
        self.graph_sequence = dg.get_graph_sequence()
        self.dataset = dataset
        self.f = f

    def eval_snapshot(self, index, emb, graph):

        lightcopyg = graph.G
        # lightcopyg.remove_edges_from(nx.selfloop_edges(lightcopyg))

        grc = GraphClusterer(lightcopyg)
        grc.detect_communities()

        embc = EmbClusterer(lightcopyg, emb)
        embc.detect_clusters()
        results = embc.print_results()

        return results

    def eval(
        self,
    ):
        N = len(self.graph_sequence)
        for i in range(N):
            try:
                print(f"snapshot number: {i}")
                s = self.eval_snapshot(i, self.embedding[i], self.graph_sequence[i])
                print(s)
                self.f.write(f"{s}\n")
            except Exception as e:
                print(f"An error occurred: {e}")
                if not self.f.closed:
                    self.f.flush()
            finally:
                if self.f and not self.f.closed:
                    self.f.flush()
        print("-------------------\n")
