"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Module defining core classes for dynamic/temporal graphs
"""

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from typing import List, Union

from tigra.core.time import determine_time_resolution

from random import sample
from statistics import mean, stdev


class Graph:
    """
    Model class for TIGRA static graphs based on networkx
    """

    def __init__(self, name, directed=True, nx_graph=None):
        """
        Parameters
        name
            graph ID
        nx_graph
            networkX graph to initialize this instance
            if nx_graph is None then an empty networkx directed graph is created
        """
        self.name = name
        if nx_graph != None:
            self.G = nx_graph if directed else nx_graph.to_undirected()
        else:
            self.G = nx.DiGraph() if directed else nx.Graph()

    def is_directed(self):
        """
        Returns boolean flag indicating wheather the graph is directed (if false then the graph is undirected)
        """
        return isinstance(self.G, nx.DiGraph)

    def add_node(self, node_id, node_label=None, node_data=None):
        """
        Adds a new node to the graph, users can provide node label and a data object attached to the node
        """
        if node_label != None and node_data != None:
            self.G.add_node(node_id, node_label=node_label, node_data=node_data)
        elif node_label != None:
            self.G.add_node(node_id, node_label=node_label)
        elif node_data != None:
            self.G.add_node(node_id, node_data=node_data)
        else:
            self.G.add_node(node_id)

    def update_node_label(self, node_id, node_label):
        """
        Updates the label of the node, use this method if nodes are created via calling add_edge
        """
        self.G.nodes[node_id]["node_label"] = node_label

    def update_node_data(self, node_id, node_vector):
        """
        Updates node data, use this method if nodes are created via calling add_edge
        """
        self.G.nodes[node_id]["node_data"] = node_vector

    def add_edge(self, src, dst, weight=None, edge_data=None):
        """
        Adds a new edge to the graph, users can provide edge weight and a data object attached to the edge
        """
        if weight != None and edge_data != None:
            self.G.add_edge(src, dst, weight=weight, edge_data=edge_data)
        elif weight != None:
            self.G.add_edge(src, dst, weight=weight)
        elif edge_data != None:
            self.G.add_edge(src, dst, edge_data=edge_data)
        else:
            self.G.add_edge(src, dst)

    def print_graph(self):
        """
        Prints all nodes and edges in the graph
        """
        print(self.name, "directed =", self.is_directed())
        print("#nodes", len(self.G.nodes()))
        for n in self.G.nodes():
            label = ""
            if "node_label" in self.G.nodes[n]:
                label = self.G.nodes[n]["node_label"]

            data = ""
            if "node_data" in self.G.nodes[n]:
                data = self.G.nodes[n]["node_data"]

            print(n, label, data)

        print("#edges", len(self.G.edges()))
        for e in self.G.edges():
            src, dst = e

            weight = ""
            if "weight" in self.G.edges[e]:
                weight = self.G.edges[e]["weight"]

            data = ""
            if "edge_data" in self.G.edges[e]:
                data = self.G.edges[e]["edge_data"]

            print(src, dst, weight, data)

    def remove_selfloop_edges(self):
        """
        Removes selfloop edges from the graph.
        """
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

    def to_undirected(self):
        """
        This method returns the undirected version of the graph.
        """
        return Graph(self.name + "-undirected", self.G.to_undirected())

    def is_empty(self):
        """
        Returns true for graphs without nodes
        """
        return self.G == None or len(self.G.nodes()) == 0

    def num_nodes(self):
        """
        Returns the number of nodes
        """
        return len(self.G.nodes())

    def num_edges(self):
        """
        Returns the number of links
        """
        return len(self.G.edges())

    def has_edge(self, src, dst):
        """
        Checks whether the graph contains edge src --> dst
        """
        return self.G.has_edge(src, dst)

    def increase_weight(self, src, dst):
        """
        increases weight of the link connecting src and dst by one
        """
        self.G.edges[src, dst]["weight"] = self.G.edges[src, dst]["weight"] + 1


class Event:
    """
    Base class for all events
    """

    def __init__(self, src, dst, timestamp, weight=None, event_data=None):
        """
        Parameters
        src
            source node
        dst
            destination node
        timestamp
            event time
        weight
            edge weight (optional)
        event data
            a data object associated to the event (optional)
        """
        self.src = src
        self.dst = dst
        self.timestamp = timestamp
        self.weight = weight
        self.event_data = event_data

    def __lt__(self, other):
        """
        < for timestamps
        """
        return self.timestamp < other.timestamp

    def __le__(self, other):
        """
        <= for timestamps
        """
        return self.timestamp <= other.timestamp

    def __str__(self):
        s = str(self.src) + "--" + str(self.dst) + ", t = " + str(self.timestamp)
        if self.weight != None:
            s += ", w = " + str(self.weight)
        if self.event_data != None:
            s += ", data = " + str(self.event_data)

        return s


class EventSequence:
    """
    Base class for all event sequences
    """

    def __init__(self, list_of_events: Union[List[Event], None] = None):
        """
        Parameters
        list_of_events
            a sequence of events for initialization
            if None an empty sequence is created
        """
        if list_of_events == None:
            self.events = []
        else:
            self.events = list_of_events
            self.events.sort()

    def add_event(self, event: Event):
        """
        Adds a new event to the sequence while perserving sorting order by timestemp
        This method should be avoided since it is extremely time inefficient,
        (EventSequence should be instantiated with a list of events)
        """
        num_events = len(self.events)
        if num_events == 0 or self.events[num_events - 1] < event:
            self.events.append(event)
        else:
            pos = 0
            while pos < num_events - 1:
                if self.events[pos] < event:
                    pos += 1
                else:
                    break

            self.events.insert(pos, event)

    def print_events(self):
        """
        Prints events in the sequence
        """
        print("Timespan", self.get_timespan())
        for e in self.events:
            print(e)

    def num_events(self):
        """
        Returns the number of events in the sequence
        """
        return len(self.events)

    def num_nodes(self):
        """
        Returns number of nodes in the whole event sequence
        """
        nodes = set()
        for event in self.events:
            nodes.add(event.src)
            nodes.add(event.dst)

        print(f"min: {min(nodes)}; max: {max(nodes)}")
        return len(nodes)

    def get_events(self):
        """
        Returns the list of events
        """
        return self.events

    def get_timespan(self):
        """
        Returns the timespan of the sequence
        """
        return self.events[-1].timestamp - self.events[0].timestamp

    def get_timespan_in_days(self):
        """
        Returns the timespan of the sequence in days
        """
        return self.get_timespan() / (24 * 60 * 60)

    def is_empty(self):
        """
        Returns true if the sequence is empty
        """
        return len(self.events) == 0

    def __getitem__(self, index):
        """
        Implementation of [] for event sequences
        """
        return self.events[index]

    def shrink_to(self, index):
        """
        shrinks event sequence to the given index
        """
        self.events = self.events[:index]


class DynamicGraph:
    """
    Base class for all dynamic graphs
    Dynamic graph is an evolutionary sequence of graphs
    """

    """
    Parameters
    list_of_graphs
        a sequence of graphs for initialization
        if None an empty sequence is created
    """

    def __init__(self, list_of_graphs: Union[List[Graph], None] = None):
        if list_of_graphs == None:
            self.graphs = []
            self.timespan_in_days = None
        else:
            self.graphs = list_of_graphs
            self.directed = False
            for g in list_of_graphs:
                if g.is_directed():
                    self.directed = True
                    break

    def init_from_event_sequence(
        self,
        event_sequence: EventSequence,
        time_resolution: str,
        directed=True,
        cumulative=False,
    ):
        """
        Creates a dynamic graph from the given sequence of events
        time_resolution can be: day, week, month, 3-months, 6-months or year
        if unknown time_resolution is provided then time_resolution will be one year

        if cumulative is set to True then snapshosts are cumulative (the next snapshots starts with the previous graph)
        """
        # time resoultion for one day
        tr = determine_time_resolution(time_resolution)

        if event_sequence.is_empty():
            return

        self.directed = directed

        id = 0
        g = Graph("t-" + str(id), directed=directed)
        start_time = event_sequence[0].timestamp
        g.add_edge(
            event_sequence[0].src,
            event_sequence[0].dst,
            event_sequence[0].weight,
            event_sequence[0].event_data,
        )

        i = 1
        while i < event_sequence.num_events():
            curr_time = event_sequence[i].timestamp
            if curr_time - start_time > tr:
                self.graphs.append(g)
                id += 1
                sid = "t-" + str(id)
                if cumulative:
                    nx_copy = nx.DiGraph(g.G) if directed else nx.Graph(g.G)
                    g = Graph(sid, nx_graph=nx_copy, directed=directed)
                else:
                    g = Graph(sid, directed=directed)

                start_time = curr_time

            g.add_edge(
                event_sequence[i].src,
                event_sequence[i].dst,
                event_sequence[i].weight,
                event_sequence[i].event_data,
            )
            i += 1

        if not g.is_empty():
            self.graphs.append(g)

        self.timespan_in_days = event_sequence.get_timespan_in_days()

    def info(self):
        """
        Print basic statistics of the dynamic graph
        """
        print("#snapshots =", len(self.graphs))
        if self.timespan_in_days != None:
            print("#timespan[days] =", self.timespan_in_days)
        s = ""
        for g in self.graphs:
            g_type = "D" if g.is_directed() else "U"
            s += (
                "["
                + g.name
                + "-"
                + str(g.num_nodes())
                + "-"
                + str(g.num_edges())
                + "-"
                + g_type
                + "]"
            )
        print(s)

    def get_graph_sequence(self):
        return self.graphs

    def is_directed(self):
        return self.directed

    def get_nx_graph_sequence(self):
        return [g.G for g in self.graphs]

    def get_nx_undirected_graph_sequence(self):
        return [g.G.to_undirected() for g in self.graphs]


class DynamicAggregateGraph:
    """
    Aggregate representation of temporal/dynamic graphs
    """

    def __init__(
        self, name, event_sequence: EventSequence, directed=True, weighted=False
    ):
        self.event_sequence = event_sequence
        self.agg_graph = Graph(name + "-agg", directed=directed)
        self.weighted = weighted
        self.form()

    def form(self):
        for event in self.event_sequence:
            src, dst = event.src, event.dst
            if not self.agg_graph.has_edge(src, dst):
                # weight = event.weight
                event_data = event.event_data
                edge_data = (event.timestamp, event_data)
                # self.agg_graph.add_edge(src, dst, weight=weight, edge_data=edge_data)

                if self.weighted:
                    self.agg_graph.add_edge(src, dst, edge_data=edge_data, weight=1)
                else:
                    self.agg_graph.add_edge(src, dst, edge_data=edge_data)
            else:
                if self.weighted:
                    self.agg_graph.increase_weight(src, dst)

    def get_aggregate_graph(self):
        return self.agg_graph

    def estimate_avg_temp_path_length(self, num_samples=100000):
        G = self.agg_graph.G
        N = list(G.nodes())
        wl = []

        for _ in range(num_samples):
            current = sample(N, 1)[0]
            current_time = 0

            visited = set()
            visited.add(current)

            walk_length = 0
            while True:
                neis = G.neighbors(current)
                cands = []
                for nei in neis:
                    time = G.edges[current, nei]["edge_data"][0]
                    if time > current_time and nei not in visited:
                        cands.append((nei, time))

                if len(cands) == 0:
                    break

                next_node = sample(cands, 1)[0]
                current = next_node[0]
                current_time = next_node[1]
                visited.add(current)

                walk_length += 1

            wl.append(walk_length)

        avg_wl = mean(wl)
        max_wl = max(wl)
        std_wl = stdev(wl)

        return avg_wl, max_wl, std_wl


class DynamicSparseMatrixGraph:
    """
    Sparse matrix representation of temporal/dynamic graphs. It doesn't use the regular Graph
    implementation for storing graphs, instead it similar to the DynamicGraph class with a
    distinction that it uses sparse matrices instead of graphs.
    """

    def __init__(
        self, name, event_sequence: EventSequence, time_resolution: str, directed=True
    ):
        print("============== DynamicSparseMatrixGraph ==============")
        number_of_nodes = event_sequence.num_nodes()
        print(number_of_nodes)

        def make_matrix():
            return lil_matrix((number_of_nodes, number_of_nodes), dtype=bool)

        tr = 24 * 60 * 60
        if time_resolution == "day":
            pass
        elif time_resolution == "week":
            tr *= 7
        elif time_resolution == "month":
            tr *= 31
        elif time_resolution == "3-months":
            tr *= 31 * 3
        elif time_resolution == "6-months":
            tr *= 31 * 6
        else:
            tr *= 31 * 12

        if event_sequence.is_empty():
            return

        self.directed = directed

        self.name = f"{name}-sm"

        self.matrices = []
        i = 0
        start_time = event_sequence[0].timestamp
        M = make_matrix()
        while i < event_sequence.num_events():
            curr_time = event_sequence[i].timestamp
            if curr_time - start_time > tr:
                self.matrices.append(M.tocsr())
                M = make_matrix()
                start_time = curr_time

            M[event_sequence[i].src - 1, event_sequence[i].dst - 1] = True
            if not directed:
                M[event_sequence[i].dst - 1, event_sequence[i].src - 1] = True
            i += 1

    def get_scipy_sparse_matrices(self):
        return self.matrices


"""
# quick tests

if __name__ == "__main__":
    events = [(1, 2, 1000), (2, 3, 500), (3, 4, 250), (7, 8, 400), (9, 10, 600), (11, 12, 100), (6, 7, 200)]
    es = EventSequence()
    for e in events:
        ev = Event(e[0], e[1], e[2])
        es.add_event(ev)

    es.print_events()

    print()
    
    events = [
        Event(1, 2, 1000), Event(1, 3, 500), Event(3, 4, 250), Event(3, 8, 400), Event(2, 3, 600), Event(4, 8, 100), Event(6, 7, 200)
    ]
    es = EventSequence(list_of_events=events)
    es.add_event(Event(2, 7, 550))
    es.add_event(Event(3, 8, 780))
    es.print_events()

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution="year")
    dg.info()

    print("Test 0")
    g = Graph("test_0")
    g.add_node("aca")
    g.add_node("mika")
    g.add_node("pera")
    g.add_edge("aca", "pera")
    g.add_edge("aca", "mika")
    g.add_edge("stojan", "gavra")
    g.add_edge("aca", "aca")
    g.print_graph()
    g.remove_selfloop_edges()
    g.print_graph()
    g = g.to_undirected()
    g.print_graph()

    print("\n\n\nTest 1")
    g = Graph("test_1")
    g.add_node("aca", node_label="partizan")
    g.add_node("mika", node_label="zvezda")
    g.add_node("pera", node_label="partizan")
    g.add_edge("aca", "pera", weight=21)
    g.add_edge("aca", "mika", weight=18)
    g.add_edge("stojan", "gavra", weight=34)
    g.print_graph()
    print()
    g.update_node_label("stojan", "zvezda")
    g.update_node_label("gavra", "partizan")
    g.print_graph()

    print("\n\n\nTest 2")
    g = Graph("test_2")
    g.add_node("aca", node_label="partizan", node_data=[12, 10, 56])
    g.add_node("mika", node_label="zvezda", node_data=[10, 20, 30])
    g.add_node("pera", node_label="partizan", node_data=[98, 87, 78])
    g.add_edge("aca", "pera", weight=21, edge_data=[1, 1])
    g.add_edge("aca", "mika", weight=18, edge_data=[10, 10])
    g.add_edge("stojan", "gavra", weight=34, edge_data=[100, 100])
    g.print_graph()
    print()
    g.update_node_label("stojan", "zvezda")
    g.update_node_data("stojan", [-1, -1, -1])
    g.update_node_label("gavra", "partizan")
    g.update_node_data("gavra", [-2, -2, -2])
    g.print_graph()
"""
