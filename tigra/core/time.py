"""
TIGRA software library

TIGRA -- Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Module dealing with time routines
"""


def determine_time_resolution(time_resolution: str):
    # time resoultion for one day
    tr = 24 * 60 * 60
    if time_resolution == "hour":
        tr = 60 * 60
    elif time_resolution == "day":
        pass
    elif time_resolution == "week":
        tr *= 7
    elif time_resolution == "month":
        tr *= 31
    elif time_resolution == "3-months":
        tr *= 31 * 3
    elif time_resolution == "6-months":
        tr *= 31 * 6
    else:
        tr *= 31 * 12

    return tr
