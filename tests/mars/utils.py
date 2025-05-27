from contextlib import contextmanager
import sys, os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def section(prefix):
    ch = "-"
    prefix = ch * 6 + " " + prefix + " "
    suffix = ch * (120 - len(prefix))
    print("\n\n\n\n" + prefix + suffix + "\n\n")


def node_labels_dict(g):
    d = dict()
    for node in g.G.nodes():
        d[node] = g.get_node_label(node)
    return d


def node_vectors_dict(g, emb):
    d = dict()
    for node in g.G.nodes():
        d[node] = emb[node]
    return d


def node_labels(g):
    return set(node_labels_dict(g).values())
