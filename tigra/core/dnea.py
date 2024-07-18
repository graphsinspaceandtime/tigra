"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

DNEA: Dynamic Network Evolution Analyser
"""

from tigra.core.dgraph import Graph, DynamicGraph
from statistics import mean

import numpy as np


class GraphComparator:
    """
    Class for comparing two consequtive graphs in an evolutionary sequence
    Determines new nodes, deleted nodes, new links and deleted links
    and computes Jaccard coefficients for nodes and links
    """

    def __init__(self, g1: Graph, g2: Graph):
        """
        g1 -- old graph
        g2 -- new graph
        """
        self.g1 = g1
        self.g2 = g2

        self.compare()

    def compare(self):
        # compare nodes
        g1_nodes = set(self.g1.G.nodes())
        g2_nodes = set(self.g2.G.nodes())
        uns = g1_nodes | g2_nodes
        ins = g1_nodes & g2_nodes
        self.jc_nodes = 0.0
        if len(uns) != 0:
            self.jc_nodes = len(ins) / len(uns)

        self.new_nodes = len(g2_nodes - g1_nodes)
        self.del_nodes = len(g1_nodes - g2_nodes)

        # compare links
        g1_links = set(self.g1.G.edges())
        g2_links = set(self.g2.G.edges())
        uls = g1_links | g2_links
        ils = g1_links & g2_links
        self.jc_links = 0.0
        if len(uls) != 0:
            self.jc_links = len(ils) / len(uls)

        self.new_links = len(g2_links - g1_links)
        self.del_links = len(g1_links - g2_links)

    def get_jc_nodes(self):
        return self.jc_nodes

    def get_jc_links(self):
        return self.jc_links

    def info(self):
        s = ""
        s += str(self.new_nodes) + ","
        s += str(self.del_nodes) + ","
        s += str(round(self.jc_nodes, 4)) + ","
        s += str(self.new_links) + ","
        s += str(self.del_links) + ","
        s += str(round(self.jc_links, 4))

        return s


class EvolDynamics:
    """
    Class providing functionalities to examine evolutionary dynamics of temporal networks
    """

    def __init__(self, dgraph: DynamicGraph):
        self.dgraph = dgraph
        self.seq = self.dgraph.get_graph_sequence()

        self.compute_jaccards()
        self.determine_nodes_activity()
        self.determine_links_activity()

    def compute_jaccards(self):
        self.msgs = []
        self.jc_nodes, self.jc_links = [], []

        for i in range(len(self.seq) - 1):
            curr_g = self.seq[i]
            next_g = self.seq[i + 1]
            gc = GraphComparator(curr_g, next_g)
            self.msgs.append(gc.info())
            self.jc_nodes.append(gc.get_jc_nodes())
            self.jc_links.append(gc.get_jc_links())

        self.avg_jc_nodes = mean(self.jc_nodes)
        self.min_jc_nodes = min(self.jc_nodes)
        self.avg_jc_links = mean(self.jc_links)
        self.min_jc_links = min(self.jc_links)

    def compute_activity_distribution(self, total_snapshots, activity_map):
        distr = np.zeros(total_snapshots + 1)
        for entry in activity_map:
            s = activity_map[entry]
            distr[len(s)] += 1.0

        total_entries = len(activity_map)

        for i in range(len(distr)):
            distr[i] /= total_entries

        return distr.tolist()

    def determine_nodes_activity(self):
        node_activity = dict()

        for i in range(len(self.seq)):
            nodes = self.seq[i].G.nodes()
            for n in nodes:
                if n in node_activity:
                    node_activity[n].append(i)
                else:
                    node_activity[n] = [i]

        self.node_activ_distr = self.compute_activity_distribution(
            len(self.seq), node_activity
        )

    def determine_links_activity(self):
        link_activity = dict()
        for i in range(len(self.seq)):
            links = self.seq[i].G.edges()
            for l in links:
                if l in link_activity:
                    link_activity[l].append(i)
                else:
                    link_activity[l] = [i]

        self.link_activ_distr = self.compute_activity_distribution(
            len(self.seq), link_activity
        )

    def info(self):
        print("NEW-NODES,DEL-NODES,JC-NODES,NEW-LINKS,DEL-LINKS,JC-LINKS")
        for m in self.msgs:
            print(m)

        print(
            "Average JC-NODES",
            round(self.avg_jc_nodes, 4),
            "min JC-NODES",
            round(self.min_jc_nodes, 4),
        )
        print(
            "Average JC-LINKS",
            round(self.avg_jc_links, 4),
            "min JC-LINKS",
            round(self.min_jc_links, 4),
        )

        print("Node activity distribution")
        avg_node_activity = 0
        for i in range(1, len(self.node_activ_distr)):
            print(i, ",", round(self.node_activ_distr[i], 4))
            avg_node_activity += self.node_activ_distr[i] * i
        print("Average node activity", avg_node_activity)

        print("Link activity distribution")
        avg_link_activity = 0
        for i in range(1, len(self.link_activ_distr)):
            print(i, ",", round(self.link_activ_distr[i], 4))
            avg_link_activity += self.link_activ_distr[i] * i
        print("Average link activity", avg_link_activity)

    def get_jc_nodes(self):
        return self.jc_nodes, self.avg_jc_nodes, self.min_jc_nodes

    def get_jc_links(self):
        return self.jc_links, self.avg_jc_links, self.min_jc_links
