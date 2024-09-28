"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Dynamic Barabasi-Albert graph generator
"""

import typing
from scipy.sparse import lil_matrix
from tigra.core.dgraph import EventSequence, Event
from tigra.generators.generator import Generator


class DBAGenerator(Generator):
    """
        n   -- the number of nodes
        m   -- the number of old nodes to which a newly created node is attached (n > m)
        m_0 -- the number of nodes in the initial graph (m <= m_0)
    """
    def __init__(self,
        n,
        m,
        m_0, 
        seed: typing.Union[int, None]=None,
        time_resolution="hour"
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
            adj_nodes = self.rng.choices(
                list(range(i)),
                weights=P,
                k=self.m
            )

            # create links
            for adj_node in adj_nodes:
                M[i, adj_node], M[adj_node, i] = True, True
                events.append(Event(i, adj_node, self.current_t))
                self.current_t += self.delta_t

        return EventSequence(events)

        


# ===============================
# Original implementation by Dule
# svc: overengineered :)
# ===============================
#
#class DBAGenerator(Generator):
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

