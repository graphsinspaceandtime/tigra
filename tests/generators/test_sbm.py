import sys
import random
import numpy as np

sys.path.append("../..")

from tigra.generators.sbm import SBMGenerator

c = [random.choice([0, 1, 2]) for _ in range(50)]
P = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

gen = SBMGenerator(c, P)

sequence = gen.generate()

print(
    "\n".join(
        [
            f"{event.src} - {event.dst} - {event.timestamp}"
            for event in sequence.get_events()
        ]
    )
)
