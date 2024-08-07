import sys
import numpy as np

sys.path.append("../..")

from tigra.core.dgraph import EventSequence
from tigra.generators.generator import Generator
from tigra.generators.sbm import SBMGenerator


class DynamicSBMGenerator(Generator):
    def __init__(self, C: np.ndarray, P: np.ndarray, directed=False):
        super().__init__()
        if C.ndim != 2:
            raise Exception("class matrix has incorrect size")

        num_class = C.shape[1]

        if (
            P.ndim != 3
            or P.shape[0] != self.T
            or P.shape[1] != num_class
            or P.shape[2] != num_class
        ):
            raise Exception("probability ndarray has incorrect size")

        self.C = C
        self.T = C.shape[0]
        self.P = P
        self.directed = directed

    def generate(self) -> EventSequence:
        es = EventSequence()

        for t in range(self.T):
            sbm = SBMGenerator(self.C[t], self.P[t], self.directed)
            for event in sbm.generate().get_events():
                es.add_event(event)

        return es
