"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Module defining functions and classes for loading datasets
"""

import re

from .dgraph import EventSequence, Event


def load_netrep_dataset(path):
    events = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("%"):
                continue
            toks = re.split(r"\s+", line)
            for i in range(len(toks)):
                toks[i] = toks[i].strip()

            src = int(toks[0])
            dst = int(toks[1])
            weight = float(toks[2])
            time = int(toks[3])

            events.append(Event(src, dst, time, weight=weight))

    events_sequence = EventSequence(list_of_events=events)

    return events_sequence


"""
# quick tests 
if __name__ == "__main__":
    es = load_netrep_dataset("../../data/netrep/ia-radoslaw-email.edges")
    print(es.num_events())

    es = load_netrep_dataset("../../data/netrep/soc-epinions-trust-dir.edges")
    print(es.num_events())
"""
