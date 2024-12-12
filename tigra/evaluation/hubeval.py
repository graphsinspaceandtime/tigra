"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

HUB-based evaluation of dynamic graph embedding methods

author: aleksandart@dmi.uns.ac.rs
"""

from tigra.core.greval import GraphReconstructionEval
from tigra.evaluation.cumstat import CumulativeCorrelation, CumulativeMWU, correlation


class HUBEval:
    def __init__(self, embedding, dg):
        self.embedding = embedding
        self.graph_sequence = dg.get_graph_sequence()
        self.f1_hub_cumul = CumulativeCorrelation()
        self.prec_hub_cumul = CumulativeCorrelation()
        self.rec_hub_cumul = CumulativeCorrelation()
        self.f1_hub_mwu = CumulativeMWU()

    def eval_snapshot(self, index, emb, graph):
        gre = GraphReconstructionEval(graph, emb, report_warnings=False)
        f1_vals = gre.get_F1_per_node()
        prec_vals = gre.get_precision_per_node()
        rec_vals = gre.get_recall_per_node()

        graph_copy = graph.G

        nodes = G.num_nodes

        total_degree = 0
        degrees = dict()

        for n in range(nodes):
            total_degree = total_degree + len(graph.get_neighbours(n))
            degrees[n] = len(graph.get_neighbours(n))
        

        self.f1_hub_cumul.add_data(f1_vals, total_degree) 
        self.prec_hub_cumul.add_data(prec_vals, total_degree)
        self.rec_hub_cumul.add_data(rec_vals, total_degree)

        degrees_sorted = dict(sorted(degrees.items(), key=lambda item: item[1]))

        self.f1_hub_mwu.add_high_low_groups_distr(degrees_sorted, f1_vals)

    def eval(self):
        N = len(self.graph_sequence)
        for i in range(N):
            s = self.eval_snapshot(i, self.embedding[i], self.graph_sequence[i])
            print(s)
        print("-------------------\n")
        print("F1\n", self.f1_nclid_cumul.get_correlation())
        print("\nPREC\n", self.prec_nclid_cumul.get_correlation())
        print("\nREC\n", self.rec_nclid_cumul.get_correlation())