import sys
import random
import numpy as np

sys.path.append("../..")

from tigra.generators.dsbm import DSBMGenerator

"""
$ C $ is a 2d array and $ c^i $ is a vector which corresponds to a row $ i $ in 2d array $ C $.
$ c $ is a vector containing class enumerations for each node in a graph. $ c_j $ is the class of
the $ j $-th node.

Each row $ i $ in $ C $ represents class memberships at time $ i $ in the dynamic graph.
"""
C = np.array([[random.choice([0, 1, 2]) for _ in range(50)] for _ in range(5)])

"""
$ P $ is a 3d array and $ P^i $ is a 2d array representing the matrix of probabilities for inter and
intra links between each class in graph at time $ i $.
"""
P = np.array([[[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]] for _ in range(5)])

gen = DSBMGenerator(C, P)

sequence = gen.generate()

print(
    "\n".join(
        [
            f"{event.src} - {event.dst} - {event.timestamp}"
            for event in sequence.get_events()
        ]
    )
)
