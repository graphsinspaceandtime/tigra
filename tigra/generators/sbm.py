import sys
import typing
import numpy as np

sys.path.append("../..")

from tigra.core.dgraph import EventSequence, Event
from tigra.generators.generator import Generator


class SBMGenerator(Generator):
    """
    Wrapper class for the NetworkX SBM graph generator.
    """

    def __init__(
        self,
        c: typing.List[int],
        P: np.ndarray,
        directed=False,
        seed: typing.Union[typing.Any, None] = None,
    ):
        super().__init__(seed)

        num_class = len(set(c))

        # probability matrix for forming edges between classes
        if P.ndim != 2 or P.shape[0] != num_class or P.shape[1] != num_class:
            raise Exception("probability matrix has incorrect size")

        self.c = c
        self.P = P
        self.directed = directed
        self.initial_timestamp = 0

    def generate(self) -> EventSequence:
        events = []

        for i in range(len(self.c)) if self.directed else range(len(self.c) - 1):
            for j in range(len(self.c)) if self.directed else range(i, len(self.c)):
                if i == j:
                    continue
                if self.rng.random() <= self.P[self.c[i], self.c[j]]:
                    events.append(Event(i, j, timestamp=self.initial_timestamp))

        return EventSequence(events)
