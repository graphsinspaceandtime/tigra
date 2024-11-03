"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Dynamic Barabasi-Albert graph generator
"""

import typing
import networkx as nx
from scipy.sparse import lil_matrix
from tigra.core.dgraph import EventSequence, Event
from tigra.generators.generator import Generator


#
# svc, dynamic scale-free networks by the Barabasi-Albert model
#
class DBAGenerator(Generator):
    """
    n   -- the number of nodes
    m   -- the number of old nodes to which a newly created node is attached (n > m)
    m_0 -- the number of nodes in the initial graph (m <= m_0)
    """

    def __init__(
        self,
        n: int,
        m: int,
        m_0: int,
        seed: typing.Union[int, None] = None,
        time_resolution="hour",
    ):
        if m >= n:
            raise Exception("DBAGenerator, m >= n")

        if m > m_0:
            raise Exception("DBAGenerator, m > m_0")

        super().__init__(seed, time_resolution)
        self.n = n
        self.m = m
        self.m_0 = m_0

    def generate(self) -> EventSequence:
        events: typing.List[Event] = []
        M = lil_matrix((self.n, self.n), dtype=bool)

        # create initial graph that is completely connected
        for j in range(1, self.m_0):
            for i in range(0, j):
                M[i, j], M[j, i] = True, True
                events.append(Event(i, j, self.current_t))
                self.current_t += self.delta_t

        # net growth
        for i in range(self.m_0, self.n):
            # compute degrees
            K = M[:i, :i].sum(axis=0).tolist()[0]

            # compute sum of degrees
            s = M[:i, :i].sum()

            # compute attachment probability
            P = [k / s for k in K]

            # attach to old nodes based on probabilities vector P
            adj_nodes = self.rng.choices(list(range(i)), weights=P, k=self.m)

            # create links
            for adj_node in adj_nodes:
                M[i, adj_node], M[adj_node, i] = True, True
                events.append(Event(i, adj_node, self.current_t))
                self.current_t += self.delta_t

        return EventSequence(events)


#
# svc, dynamic scale-free networks with communities
#
class DBACGenerator(Generator):
    """
    NC
        the number of nodes per community/cluster
    m_in
        the number of old nodes in the same cluster to which a newly created node is attached
    m_out
        the number of old nodes outside the cluster to which a newly created node is attached
    m_0
        the initial number of nodes per cluster (m <= m_0)
    """

    def __init__(
        self,
        NC: typing.List[int],
        m_in: int,
        m_out: int,
        m_0: int,
        time_resolution="hour",
        seed: typing.Union[typing.Any, None] = None,
    ):
        self.total_attachment_links = m_in + m_out
        self.total_nodes = sum(NC)

        if self.total_attachment_links >= self.total_nodes:
            raise Exception("DBACGenerator, m_in + m_out >= total nodes")

        if self.total_attachment_links > m_0:
            raise Exception("DBACGenerator, m_in + m_out > m_0")

        for nc in NC:
            if m_0 > nc:
                raise Exception("DBACGenerator, m_0 > nc, invalid NC")

        super().__init__(seed, time_resolution)
        self.NC = NC
        self.m_in = m_in
        self.m_out = m_out
        self.m_0 = m_0

    def init(self):
        initial_communities = []
        self.community_map = dict()
        current_id, community_id = 0, 0
        for nc in self.NC:
            init = list(range(current_id, current_id + self.m_0))
            initial_communities.append(init)
            next_id = current_id + nc
            for i in range(current_id, next_id):
                self.community_map[i] = community_id

            community_id += 1
            current_id = next_id

        self.all_nodes = set(range(0, self.total_nodes))

        # create initial chain of cliques
        for c in initial_communities:
            for n in c:
                self.G.add_node(n)
                self.all_nodes.remove(n)

            for j in range(c[1], c[-1] + 1):
                for i in range(c[0], j):
                    self.G.add_edge(i, j)
                    event_data_str = (
                        str(self.community_map[i]) + "," + str(self.community_map[j])
                    )
                    self.events.append(
                        Event(i, j, self.current_t, event_data=event_data_str)
                    )
                    self.current_t += self.delta_t

        for i in range(len(initial_communities) - 1):
            n1 = initial_communities[i][0]
            n2 = initial_communities[i + 1][0]
            self.G.add_edge(n1, n2)
            event_data_str = (
                str(self.community_map[n1]) + "," + str(self.community_map[n2])
            )
            self.events.append(Event(n1, n2, self.current_t, event_data=event_data_str))
            self.current_t += self.delta_t

    def generate(self) -> EventSequence:
        self.events: typing.List[Event] = []
        self.G = nx.Graph()

        self.init()

        nodes_to_add = list(self.all_nodes)
        # self.rng.shuffle(nodes_to_add)

        # print(len(self.G.nodes()))
        # print(len(nodes_to_add))

        for new_node in nodes_to_add:
            old_nodes = self.G.nodes()

            in_nodes, out_nodes = [], []
            in_degs, out_degs = [], []
            total_in, total_out = 0, 0

            for old_node in old_nodes:
                deg_old = self.G.degree(old_node)
                if self.community_map[old_node] == self.community_map[new_node]:
                    in_nodes.append(old_node)
                    in_degs.append(deg_old)
                    total_in += deg_old
                else:
                    out_nodes.append(old_node)
                    out_degs.append(deg_old)
                    total_out += deg_old

            P_in = [k / total_in for k in in_degs]
            P_out = [k / total_out for k in out_degs]

            self.G.add_node(new_node)

            adj_nodes_in = self.rng.choices(in_nodes, weights=P_in, k=self.m_in)

            for adj_node in adj_nodes_in:
                self.G.add_edge(adj_node, new_node)
                event_data_str = (
                    str(self.community_map[adj_node])
                    + ","
                    + str(self.community_map[new_node])
                )
                self.events.append(
                    Event(adj_node, new_node, self.current_t, event_data=event_data_str)
                )
                self.current_t += self.delta_t

            adj_nodes_out = self.rng.choices(out_nodes, weights=P_out, k=self.m_out)

            for adj_node in adj_nodes_out:
                self.G.add_edge(adj_node, new_node)
                event_data_str = (
                    str(self.community_map[adj_node])
                    + ","
                    + str(self.community_map[new_node])
                )
                self.events.append(
                    Event(adj_node, new_node, self.current_t, event_data=event_data_str)
                )
                self.current_t += self.delta_t

        return EventSequence(self.events)


# ===============================
# Original implementation by Dule
# svc: overengineered :)
# ===============================
#
# class DBAGenerator(Generator):
#    """
#    A generator implementation for dynamic Barabasi-Albert model.
#
#    Arguments:
#        n: list of numbers of new nodes for each timestamp
#        m: list of numbers of new links for each timestamp (per new node)
#        initial_graph: instance of an initial NetworkX graph (default is a star graph with m+1
#                       nodes)
#        number_of_nodes: number of distinct nodes that are in total recorded in this temporal graph
#        directed: is graph directed or not (default is False if initial_graph is not specified and
#                  the directionality of the initial_graph if it is specified)
#        seed: seed for the random number generator
#
#    Either initial_graph is specified which provides information for both number_of_nodes and
#    directed parameters or number_of_nodes and directed is specified, both should not be specified
#    at the same time.
#    """
#
#    def __init__(
#        self,
#        n: typing.List[int],
#        m: typing.List[int],
#        initial_graph: typing.Union[nx.Graph, nx.DiGraph, None] = None,
#        number_of_nodes: typing.Union[int, None] = None,
#        directed: typing.Union[bool, None] = None,
#        seed: typing.Union[int, None] = None,
#    ):
#        super().__init__(seed)
#        # TODO: check arguments
#        if initial_graph == None:
#            assert number_of_nodes != None
#            assert directed != None
#            # TODO: Is this okay?
#            initial_graph = nx.star_graph(
#                m[0] + 1, nx.DiGraph if directed else nx.Graph
#            )
#        else:
#            number_of_nodes = initial_graph.number_of_nodes()
#            directed = initial_graph.is_directed()
#
#        events: typing.List[Event] = []
#        max_number_of_nodes = number_of_nodes + sum(n)
#        M = lil_matrix((max_number_of_nodes, max_number_of_nodes), dtype=bool)
#        for i, j in initial_graph.edges:  # type: ignore
#            M[i, j] = True
#            events.append(Event(i, j, 0))
#
#        self.n = n
#        self.m = m
#        self.T = len(n)
#        self.events = events
#        self.M = M
#        self._initial_number_of_nodes = number_of_nodes
#
#    def generate(self) -> EventSequence:
#        i_offset = self._initial_number_of_nodes
#        # for each timestamp
#        for t in range(self.T):
#            # create a defined number of nodes
#            for n in range(i_offset, i_offset + self.n[t]):
#                # calculate degree of each existing node (not taking into account newly added ones)
#                K = self.M[:i_offset, :i_offset].sum(axis=0).tolist()[0]  # type: ignore
#                # calculate sum of degrees of all existing nodes
#                s = self.M[:i_offset, :i_offset].sum()  # type: ignore
#                # calculate probability for each node being selected
#                P = [k / s for k in K]
#                # pick predefined number of nodes based on probabilities vector P
#                adj_nodes = self.rng.choices(
#                    list(range(i_offset)),
#                    weights=P,
#                    k=self.m[t],
#                )
#                # add new nodes
#                for adj_node in adj_nodes:
#                    self.M[n, adj_node] = True
#                    self.events.append(Event(n, adj_node, t))
#            # increment the offset value
#            i_offset += self.n[t]
#
#        return EventSequence(self.events)
