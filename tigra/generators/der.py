"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Dynamic Erdos-Renyi graph generator
"""

import typing
from tigra.core.dgraph import EventSequence, Event
from tigra.generators.generator import Generator

class DERGenerator(Generator):
    """
        n -- the number of nodes
        p -- connection probability for each pair of nodes
    """
    def __init__(
        self,
        n,
        p,
        seed: typing.Union[int, None]=None,
        time_resolution="hour"
    ):
        super().__init__(seed, time_resolution)
        self.n = n
        self.p = p


    def generate(self) -> EventSequence:
        events: typing.List[Event] = []

        for j in range(1, self.n):
            for i in range(0, j):
                if self.rng.random() <= self.p:
                    events.append(Event(i, j, self.current_t))
                    self.current_t += self.delta_t
                    
        return EventSequence(events)