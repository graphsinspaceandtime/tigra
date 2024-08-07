import sys
import typing
import numpy as np
from scipy.sparse import lil_matrix

sys.path.append("../..")

from tigra.core.dgraph import EventSequence, Event
from tigra.generators.generator import Generator
from tigra.generators.sbm import SBMGenerator


class SBTMGenerator(Generator):
    """
    A generator implementation for stochastic block transition model defined in Xu, K., 2015,
    February. Stochastic block transition models for dynamic networks. In Artificial Intelligence
    and Statistics (pp. 1079-1087). PMLR.

    Arguments:
        C: matrix containing class memberships of nodes across units of time
        P: 3-d array of probabilities for new links between classes across units of time
        P_exist: 3-d array of probabilities for retention of old links between classes across units
                 time. P_exist[:,:,0] is ignored
        directed: is graph directed or not
    """

    def __init__(
        self,
        C: np.ndarray,
        P: np.ndarray,
        P_exist: np.ndarray,
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
        if (
            P_exist.ndim != 3
            or P_exist.shape[0] != T
            or P_exist.shape[1] != num_class
            or P_exist.shape[2] != num_class
        ):
            raise Exception("existing probability ndarray has incorrect size")

        self.C = C
        self.T = T
        self.P = P
        self.P_exist = P_exist
        self.directed = directed

    def generate(self) -> EventSequence:
        events: typing.List[Event] = []
        num_nodes = len(set(self.C[0]))
        M = lil_matrix((num_nodes, num_nodes), dtype=bool)

        sbm = SBMGenerator(self.C[0], self.P[0], self.directed)
        for event in sbm.generate():
            M[event.src, event.dst] = True
            events.append(event)

        for t in range(1, self.T):
            for i in range(num_nodes) if self.directed else range(num_nodes - 1):
                for j in range(num_nodes) if self.directed else range(i, num_nodes):
                    if i == j:
                        continue  # no self loops

                    r = self.rng.random()

                    if M[i, j]:
                        if r > self.P_exist[t, self.C[t, i], self.C[t, j]]:
                            M[i, j] = False
                    else:
                        if r <= self.P[t, self.C[t, i], self.C[t, j]]:
                            M[i, j] = True

                    if M[i, j]:
                        events.append(Event(i, j, t))

        return EventSequence(events)
