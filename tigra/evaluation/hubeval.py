"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

HUB-based evaluation of dynamic graph embedding methods

author: aleksandart@dmi.uns.ac.rs
"""

from tigra.core.greval import GraphReconstructionEval
from tigra.evaluation.cumstat import CumulativeCorrelation, CumulativeMWU, correlation
from tigra.core.dgraph import DynamicAggregateGraph


class HUBEval:
    def __init__(self, embedding, dg):
        self.embedding = embedding
        if isinstance(dg, DynamicAggregateGraph):
            self.graph_sequence = [dg]
        elif len(embedding.embeddings) < len(dg.get_graph_sequence()):
            n = len(dg.get_graph_sequence())
            k = len(embedding.embeddings)
            self.graph_sequence = dg.get_graph_sequence()[n - k : n]
        else:
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

        # graph_copy = graph.G.to_undirected()
        # nodes = graph_copy.number_of_nodes()
        # graph_dg_copy = self.dg

        total_degree = 0
        degrees = dict()

        # for n in range(nodes):
        for n in graph.G.nodes():
            deg = graph.G.degree(n)
            total_degree = total_degree + deg  # len(graph_dg_copy.get_neighbours(n))
            degrees[n] = deg  # len(graph_dg_copy.get_neighbours(n))

        self.f1_hub_cumul.add_data(f1_vals, degrees)
        self.prec_hub_cumul.add_data(prec_vals, degrees)
        self.rec_hub_cumul.add_data(rec_vals, degrees)

        degrees_sorted = dict(
            sorted(degrees.items(), reverse=True, key=lambda item: item[1])
        )

        # print(degrees_sorted)

        self.f1_hub_mwu.add_high_low_groups_distr(degrees_sorted, f1_vals)

        s = str(index + 1) + ","
        s += str(round(gre.getF1(), 4)) + ","
        s += str(round(gre.get_avg_precision(), 4)) + ","
        s += str(round(gre.get_avg_recall(), 4)) + ","
        s += str(round(correlation(f1_vals, degrees), 4))

        return s

    def eval(self):
        N = len(self.graph_sequence)
        print("#snap, F1, prec, rec, correlation(F1, degree)")
        for i in range(N):
            s = self.eval_snapshot(i, self.embedding[i], self.graph_sequence[i])
            print(s)
        print("\n[cummulative correlations] -------------------")
        print("F1", self.f1_hub_cumul.get_correlation())
        print("PREC", self.prec_hub_cumul.get_correlation())
        print("REC", self.rec_hub_cumul.get_correlation())
        print("--- MWU")
        print(
            "AVG-F1(hubs), AVG-F1(non-hubs), #hubs, #non-hubs, U, p(U), ACC, PS(hubs), PS(non-hubs)"
        )
        print(self.f1_hub_mwu.mwu())
