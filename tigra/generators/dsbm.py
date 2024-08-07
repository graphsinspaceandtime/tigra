import sys
import typing
import numpy as np

sys.path.append("../..")

from tigra.core.dgraph import EventSequence
from tigra.generators.generator import Generator
from tigra.generators.sbm import SBMGenerator


class DynamicSBMGenerator(Generator):
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

        T, num_class = C.shape

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
