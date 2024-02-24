"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Module defining functions and classes for loading datasets
"""

import re

from .dgraph import EventSequence, Event


def load_netrep_dataset(path, has_weight=True) -> EventSequence:
    events = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("%"):
                continue
            # split at nondigit characters
            # TODO: will break on float data
            toks = re.split(r"\D+", line)
            for i in range(len(toks)):
                toks[i] = toks[i].strip()

            src = int(toks[0])
            dst = int(toks[1])
            next_index = 2
            weight = None
            if has_weight:
                weight = float(toks[2])
                next_index += 1
            time = int(toks[next_index])

            events.append(Event(src, dst, time, weight=weight))

    events_sequence = EventSequence(list_of_events=events)

    return events_sequence


def load_snap_dataset(path, has_weight=False) -> EventSequence:
    events = []

    with open(path) as file:
        for line in file:

            line = line.strip()

            tokens = re.split(r"\D+", line)
            for i in range(len(tokens)):
                tokens[i] = tokens[i].strip()
            
            src = int(tokens[0])
            tgt = int(tokens[1])
            time = int(tokens[2])

            events.append(Event(src=src, dst=tgt, timestamp=time))


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
