"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Generators for dynamic graphs with communities/clusters
"""

import sys
import typing
import numpy as np

sys.path.append("../..")

from tigra.core.dgraph import EventSequence, Event
from tigra.generators.generator import Generator
from tigra.generators.sbm import SBMGenerator


#
# svc, dynamic planted partition generator
#
class DPPGenerator(Generator):
    """
    NC
        the number of nodes per community/cluster
    p
        intra-community link probability
    q
        inter-community link probability
    """

    def __init__(
        self,
        NC: typing.List[int],
        p: float,
        q: float,
        time_resolution="hour",
        seed: typing.Union[typing.Any, None] = None,
    ):
        super().__init__(seed, time_resolution)

        self.num_communities = len(NC)
        if self.num_communities < 2:
            raise Exception(
                "invalid number of communities/clusters, must be >= 2 by specifing NC"
            )

        self.total_nodes = sum(NC)

        self.NC = NC
        self.p = p
        self.q = q

        self.community_id = list()
        cum = 0
        for nc in NC:
            cum += nc
            self.community_id.append(cum)

    def get_community_id(self, node_id):
        counter = 0
        for cid in self.community_id:
            if node_id < cid:
                return counter
            counter += 1

    def generate(self) -> EventSequence:
        events: typing.List[Event] = []

        pairs = []
        for j in range(1, self.total_nodes):
            for i in range(0, j):
                pairs.append((i, j))

        self.rng.shuffle(pairs)

        for pair in pairs:
            src, dst = pair[0], pair[1]
            src_c, dst_c = self.get_community_id(src), self.get_community_id(dst)
            same_community = src_c == dst_c
            cp = self.p if same_community else self.q
            if self.rng.random() <= cp:
                event_data_str = str(src_c) + "," + str(dst_c)
                events.append(
                    Event(pair[0], pair[1], self.current_t, event_data=event_data_str)
                )
                self.current_t += self.delta_t

        return EventSequence(events)


#
# dule, full parametrized stochastic block model
# TODO: proper timestamps
#
class DSBMGenerator(Generator):
    def __init__(
        self,
        C: np.ndarray,
        P: np.ndarray,
        directed=False,
        seed: typing.Union[typing.Any, None] = None,
    ):
        super().__init__(seed)
        if C.ndim != 2:
            raise Exception("class matrix has incorrect size")

        T, _ = C.shape
        num_class = len(set(C[0, :]))

        if (
            P.ndim != 3
            or P.shape[0] != T
            or P.shape[1] != num_class
            or P.shape[2] != num_class
        ):
            raise Exception("probability ndarray has incorrect size")

        self.C = C
        self.T = T
        self.P = P
        self.directed = directed

    def generate(self) -> EventSequence:
        events = []

        for t in range(self.T):
            sbm = SBMGenerator(self.C[t], self.P[t], self.directed)
            sbm.rng = self.rng
            for event in sbm.generate().get_events():
                events.append(event)

        return EventSequence(events)
