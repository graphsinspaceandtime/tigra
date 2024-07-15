import sys
from abc import ABC, abstractmethod

sys.path.append("../..")

from tigra.core.dgraph import EventSequence


class Generator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self) -> EventSequence:
        """
        Generates an EventSequence based on the Generator parameters.
        """
        pass
