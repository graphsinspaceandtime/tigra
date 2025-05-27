import sys

from tigra.core.loaders import load_snap_dataset
from tigra.embeddings.rndw.rndwbase import DynN2V
from tigra.generators.dsbm import DPPGenerator

sys.path.append("../..")


from tigra.core.dgraph import DynamicGraph
from utils import *
from data.bank import Databank


def do_dg_embs(dg, embs):
    dg.info()
    gs = dg.get_graph_sequence()
    for i in range(len(gs)):
        g = gs[i]
        e = embs[i]
        labels_str = f"{node_labels(g)}" if g.labeled else "NOT LABELED"
        print(f"{i}: {g.num_nodes()} {g.num_edges()} {labels_str}")


def do_real(db, dataset_name):
    section(dataset_name)

    es = db.get_data(dataset_name)
    metadata = db.get_metadata(dataset_name)

    print(metadata)
    print()

    dg = DynamicGraph()
    dg.init_from_event_sequence(
        es,
        time_resolution=metadata["resolution"],
        directed=False,
        cumulative=metadata["cumulative"],
    )

    dn2v = DynN2V(200, 0.5, 4)
    dn2v.make_embedding(dg)

    do_dg_embs(dg, dn2v.embeddings)


def do_reals():
    db = Databank("../../data")

    # for dataset_name in db.get_keys():
    for dataset_name in [
        "ia-hospital-ward-proximity-attr",
        "ia-primary-school-proximity-attr",
    ]:
        do_real(db, dataset_name)


def do_synthetics():
    section("synthetic")

    # dppg = DPPGenerator([500, 250, 150, 100], 0.03, 0.001)
    # dppg.generate_and_save("../dpp_1000_4.events")
    # es = load_snap_dataset("../dpp_1000_4.events")
    # es = load_snap_dataset("../../data/synthetic/dbac_1000_4.events")

    dg = DynamicGraph()
    dg.init_from_event_sequence(
        es, time_resolution="month", directed=False, cumulative=True
    )

    dn2v = DynN2V(100, 0.5, 4)
    dn2v.make_embedding(dg)

    do_dg_embs(dg, dn2v.embeddings)


def do():
    # do_reals()
    do_synthetics()


if __name__ == "__main__":
    do()
