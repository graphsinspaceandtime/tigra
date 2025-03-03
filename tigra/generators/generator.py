"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Base class for dynamic graph generators
"""

from abc import ABC, abstractmethod
import random
from tigra.core.dgraph import EventSequence
from tigra.core.time import determine_time_resolution


class Generator(ABC):
    def __init__(self, seed, time_resolution="hour"):
        self.rng = random.Random(seed)
        self.delta_t = determine_time_resolution(time_resolution)
        self.current_t = 0  #  January 1, 1970 12:00:00 AM

    @abstractmethod
    def generate(self) -> EventSequence:
        """
        Generates an EventSequence based on the Generator parameters.
        """
        pass

    def generate_and_save(self, file_name):
        es = self.generate()
        print("#total events generated", es.num_events())
        f = open(file_name, "w")
        for e in es:
            s = str(e.src) + "," + str(e.dst) + "," + str(e.timestamp)
            ed = e.event_data
            if ed != None:
                src_class = ed["src_class"]
                dst_class = ed["dst_class"]
                s += "," + str(src_class) + "," + str(dst_class)
            s += "\n"

            f.write(s)

        f.close()
        print("#events saved to", file_name)
