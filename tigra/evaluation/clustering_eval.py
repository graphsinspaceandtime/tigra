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

        # print(num_com)

        n_clusters = []
        if num_com != None:
            ts = set(num_com)
            for n in ts:
                n_clusters.append(n)

        # default_k = [2, 3, 4, 5, 10]
        # for n in default_k:
        #     if n not in n_clusters:
        #        n_clusters.append(n)

        # print(n_clusters)

        self.grid_parms = {
            KMeans: {"n_clusters": n_clusters},
            DBSCAN: {
                "eps": [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 5, 10, 50, 100, 200, 500],
                "min_samples": [3, 5, 7, 10],
            },
            AgglomerativeClustering: {
                "n_clusters": n_clusters,
                "linkage": ["ward", "complete", "average"],
            },
        }

        self.clustering_algorithms = [
            (KMeans(random_state=42, n_init=10), self.grid_parms[KMeans]),
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
            # k = grid_search.best_params_["n_clusters"]

            labels = getattr(best_alg, "labels_", None)
            if labels is None and hasattr(best_alg, "fit_predict"):
                labels = best_alg.fit_predict(self.X)

            # print(labels)
            lab = set(labels)
            num_labs = len(lab)
            part = []
            for _ in range(num_labs):
                part.append([])

            for i in range(len(self.nodes_list)):
                node = self.nodes_list[i]
                cl = labels[i]
                part[cl].append(node)

            # print(part)

            mod = modularity(self.graph, part)
            if labels is not None:
                if alg_name == "DBSCAN":
                    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                else:
                    num_clusters = len(set(labels))
            else:
                num_clusters = "N/A"

            # modularnost

            self.results.append(
                {
                    "alg": alg_name,
                    "communities": num_clusters,
                    "params": grid_search.best_params_,
                    "silhouette_score": grid_search.best_score_,
                    "labels": labels,
                    "modularity": mod,
                }
            )

    def score_fn(self, estimator, X):
        labels = estimator.fit_predict(X)
        # print(f"sil score labels no:{len(set(labels))}, {len(set(estimator.labels_))}")
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
        communities_list = []
        for com in communities_generator:
            communities_list.append(com)

        # print(communities_generator)
        # first_level_communities = next(communities_generator)
        # communities_list = [list(c) for c in first_level_communities]
        return communities_list

    def get_lp(self, graph):
        iter = asyn_lpa_communities(graph, seed=0)
        part = []
        for c in iter:
            part.append(c)

        # print(part)

        return part

    def get_gmc(self, graph):
        return np.asarray(greedy_modularity_communities(graph))

    def detect_communities(self):
        algorithms = [
            ("lp", self.get_lp),
            ("Louvain", louvain_communities),
            ("GMO", self.get_gmc),
        ]

        for name, method in algorithms:
            communities = method(self.G)
            mod = modularity(self.G, communities)
            # silhouette, labels = self.silhouette(communities)
            # print("Silhouette Score:", silhouette)

            self.results.append(
                {
                    "alg": name,
                    "communities": communities,
                    "modularity": mod,
                    # "silhouette_score": silhouette,
                    # "labels": labels,
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

            # print(s)

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

        nc = []
        for ginfo in grc.results:
            k = ginfo["communities"]
            nc.append(len(k))

        embc = EmbClusterer(lightcopyg, emb, num_com=nc)
        embc.detect_clusters()

        return grc.results, embc.results

    def eval(
        self,
    ):
        N = len(self.graph_sequence)
        algorithms = ["lp", "Louvain", "GMO", "KMeans", "DBSCAN", "AgglomerativeClustering"]

        for i in range(N):
            try:
                print(f"snapshot number: {i}")
                grc, embc = self.eval_snapshot(
                    i, self.embedding[i], self.graph_sequence[i]
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
                        
                    if einfo:
                        row.extend([einfo["modularity"], einfo["communities"]])

                    if alg in ["KMeans", "DBSCAN", "AgglomerativeClustering"]:
                        row.append(einfo["silhouette_score"])

                self.save_results_to_csv( row)
            except Exception as e:
                print(f"An error occurred: {e}")
        print("-------------------\n")

    def save_results_to_csv(self, row):
        with open(self.path, "a", encoding="utf-8") as f:

            output = ", ".join(map(str, row))

            f.write(f"{output}\n")
            print(f"{output}")
            print(f"Results saved to file")
