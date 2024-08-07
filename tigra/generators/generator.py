import sys
from abc import ABC, abstractmethod
import random

sys.path.append("../..")

from tigra.core.dgraph import EventSequence


class Generator(ABC):
    def __init__(self, seed):
        self.rng = random.Random(seed)

    @abstractmethod
    def generate(self) -> EventSequence:
        """
        Generates an EventSequence based on the Generator parameters.
        """
        pass
