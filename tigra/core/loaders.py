"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Module defining functions and classes for loading datasets
"""

import re

from .dgraph import EventSequence, Event
from .time import determine_time_resolution

from data.bank import Databank


datasets = {
    "radoslaw-email"                   : ("month", 0),
    "fb-forum"                         : ("month", -1),
    "fb-messages"                      : ("month", 0),
    "ia-contacts_hypertext2009"        : ("day", 0),
    "ia-enron-employees"               : ("3-months", -2),
    "ia-hospital-ward-proximity-attr"  : ("day", -1),
    "ia-primary-school-proximity-attr" : ("day", 0),
    "ia-reality-call"                  : ("week", -1),
    "CollegeMsg"                       : ("month", -1),
    "sx-askubuntu"                     : ("3-months", 0),
    "sx-mathoverflow"                  : ("3-months", 0)
}


def load_event_sequence(dataset_name: str):
    if dataset_name not in datasets:
        print(f"[loaders] {dataset_name} not supported")
        return

    databank = Databank()
    es = databank.get_data(dataset_name)
    
    resol, shrink = datasets[dataset_name]

    if shrink == 0:
        return es, resol

    tr = determine_time_resolution(resol)
    timespan = es.get_timespan()
    snaps = timespan / tr
    if int(snaps) != snaps:
        snaps += 1
    snaps = int(snaps)
    
    snaps_to_keep = snaps + shrink
    time_to_keep = es.events[0].timestamp + snaps_to_keep * tr
    
    i = len(es.events) - 1
    while i >= 0 and es.events[i].timestamp > time_to_keep:
        i -= 1
 
    i += 1
    es.shrink_to(i)

    return es, resol
    



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
