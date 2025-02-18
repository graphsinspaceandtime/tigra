"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Cluster based evaluation of dynamic graph embedding methods

author: jelab@dmi.uns.ac.rs
"""

import numpy as np
import networkx as nx


from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity, coverage
from networkx.algorithms.community import girvan_newman, louvain_communities

from tigra.core.dgraph import DynamicAggregateGraph


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

        self.results = []

    def detect_clusters(self):

        for alg, grid_param in self.clustering_algorithms:
            alg_name = alg.__class__.__name__

            grid_search = GridSearchCV(
                estimator=alg, param_grid=grid_param, scoring=self.score_fn, cv=3
            )
            grid_search.fit(self.X)

            best_alg = grid_search.best_estimator_

            labels = getattr(best_alg, "labels_", None)
            if labels is None and hasattr(best_alg, "fit_predict"):
                labels = best_alg.fit_predict(self.X)

            if labels is not None:
                if alg_name == "DBSCAN":
                    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                else:
                    num_clusters = len(set(labels))
            else:
                num_clusters = "N/A"

            self.results.append(
                {
                    "alg": alg_name,
                    "communities": num_clusters,
                    "params": grid_search.best_params_,
                    "silhouette_score": grid_search.best_score_,
                    "labels": labels,
                }
            )

    def score_fn(self, estimator, X):
        labels = estimator.fit_predict(X)
        print(f"sil score labels no:{len(set(labels))}, {len(set(estimator.labels_))}")
        if len(set(labels)) <= 1:
            return -1
        return silhouette_score(X, labels)

    def print_results(self):
        results = ""
        for info in self.results:
            s = f"Algorithm: {info['alg']}"
            s += f"  Communities: {info['communities']}"
            s += f"  Params: {info['params']}"
            s += f"  Silhouette Score: {info['silhouette_score']:.3f} \n"

            print(s)

            results += s

        return results


class GraphClusterer:
    def __init__(self, graph):
        self.graph = graph
        self.G = self.graph.to_undirected()
        self.node_to_index = {node: idx for idx, node in enumerate(self.G.nodes())}

        self.results = []

    def get_gnc(self, graph):
        communities_generator = girvan_newman(graph)
        first_level_communities = next(communities_generator)
        communities_list = [list(c) for c in first_level_communities]
        return communities_list

    def get_gmc(self, graph):
        return np.asarray(greedy_modularity_communities(graph))

    def detect_communities(self):
        algorithms = [
            ("Girvan-Newman", self.get_gnc),
            ("Louvain", louvain_communities),
            ("GMO", self.get_gmc),
        ]

        for name, method in algorithms:
            communities = method(self.G)
            mod = modularity(self.G, communities)
            silhouette, labels = self.silhouette(communities)
            print("Silhouette Score:", silhouette)

            self.results.append(
                {
                    "alg": name,
                    "communities": communities,
                    "modularity": mod,
                    "silhouette_score": silhouette,
                    "labels": labels,
                }
            )

        return self.results

    def silhouette(self, communities):
        node_to_index = {node: idx for idx, node in enumerate(self.G.nodes())}

        labels = np.zeros(self.G.number_of_nodes(), dtype=int)
        for community_id, community in enumerate(communities):
            for node in community:
                labels[node_to_index[node]] = community_id

        X = nx.to_numpy_array(self.G)
        if len(communities) < 2:
            print("Nedovoljno zajednica za računanje silhouette score-a.")
            score = 0
        else:
            score = silhouette_score(X, labels)
        return score, labels

    def print_results(self):
        results = ""
        for info in self.results:
            s = f"Algorithm: {info['alg']}"
            s += f"  Communities: {info['communities']}"
            s += f"  Modularity: {info['modularity']}"
            s += f"  Silhouette Score: {info['silhouette_score']:.3f} \n"

            print(s)

            results += s

        return results


class ClusteringEval:
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
        # lightcopyg.remove_edges_from(nx.selfloop_edges(lightcopyg))

        grc = GraphClusterer(lightcopyg)
        grc.detect_communities()

        embc = EmbClusterer(lightcopyg, emb)
        embc.detect_clusters()

        return grc.results, embc.results

    def eval(
        self,
    ):
        N = len(self.graph_sequence)
        for i in range(N):
            try:
                print(f"snapshot number: {i}")
                grc, embc = self.eval_snapshot(
                    i, self.embedding[i], self.graph_sequence[i]
                )
                for ginfo in grc:
                    grc_labels = ginfo["labels"]
                    for einfo in embc:
                        emb_labels = einfo["labels"]
                        nmi_gt_community_rg = normalized_mutual_info_score(
                            grc_labels, emb_labels
                        )
                        self.save_results_to_csv(i, ginfo, einfo, nmi_gt_community_rg)
            except Exception as e:
                print(f"An error occurred: {e}")
        print("-------------------\n")

    def save_results_to_csv(self, snapshot_num, grc, embc, nmi):
        with open(self.path, "a", encoding="utf-8") as f:
            row = [
                snapshot_num,
                grc["alg"],
                embc["alg"],
                len(grc["communities"]),
                embc["communities"],
                grc["modularity"],
                grc["silhouette_score"],
                grc["silhouette_score"],
                nmi,
            ]
            output = ", ".join(map(str, row))

            f.write(f"{output}\n")
            print(f"{output}")
            print(f"Results saved to file")
