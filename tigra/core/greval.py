"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Main base class for evaluation of graph reconstruction from embedding
"""

from tigra.embeddings.embedding import SingletonEmbedding
from tigra.core.dgraph import Graph


class GraphReconstructionEval:

    def __init__(self, graph: Graph, embedding: SingletonEmbedding):
        self.graph = graph.G.to_undirected()
        self.embedding = embedding

        self.avg_precision = 0.0
        self.avg_recall = 0.0
        self.precision_per_node = dict()
        self.recall_per_node = dict()
        self.__calculate_prec_rec()

    def __calculate_prec_rec(self):
        k = self.graph.number_of_edges()
        reconstructed_graph = self.embedding.reconstruct(k, "REC").G

        if self.graph.number_of_nodes() != reconstructed_graph.number_of_nodes():
            print(
                "[WARNING, greval] original and reconstructed graph have different number of nodes"
            )

        if self.graph.number_of_edges() != reconstructed_graph.number_of_edges():
            print(
                "[WARNING, greval] original and reconstructed graph have different number of links"
            )

        s_rec = 0
        s_prec = 0
        nodes_cnt = 0
        cnt = 0

        for node in self.graph.nodes:
            if not node in reconstructed_graph.nodes():
                print("[WARNING, greval] Node", node, "skipped")
                continue

            predicted_edges = reconstructed_graph.edges(node)

            precision, recall = 0, 0

            if len(predicted_edges) != 0:
                real_edges = self.graph.edges(node)
                num_real_edges = len(real_edges)

                if num_real_edges != 0:
                    node_s = 0
                    for p_edge in predicted_edges:
                        if p_edge in real_edges:
                            node_s += 1
                            cnt += 1

                    recall = node_s / num_real_edges
                    precision = cnt / len(self.graph.edges)

            nodes_cnt += 1
            self.recall_per_node[node] = recall
            self.precision_per_node[node] = precision
            s_rec += recall
            s_prec += precision

        self.avg_precision = s_prec / nodes_cnt
        self.avg_recall = s_rec / nodes_cnt

    def get_precision_per_node(self):
        return self.precision_per_node

    def get_recall_per_node(self):
        return self.recall_per_node

    def get_avg_precision(self):
        return self.avg_precision

    def get_avg_recall(self):
        return self.avg_recall

    def getF1(self):
        p = self.get_avg_precision()
        r = self.get_avg_recall()
        if p + r == 0.0:
            return 0.0
        return 2 * p * r / (p + r)
