from tigra.data.loaders import get_dataset_by_id
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.rndw.rndwbase import DynN2V

import os

if not os.path.isdir("embs_dn2v"):
    os.mkdir("embs_dn2v")

configs = []

with open("../../../DN2V_SVC.conf", "r", encoding="utf-8") as f:
    f.readline()  # heading
    for line in f.readlines():
        config = line.split(",")
        configs.append(
            {
                "dataset": config[0],
                "id": config[0].replace("-", "_"),
                "dim": int(config[1]),
                "p": float(config[2]),
                "q": float(config[3]),
                "wl": int(config[4]),
                "nw": int(config[5]),
            }
        )
        if configs[-1]["id"] == "CollegeMsg":
            configs[-1]["id"] = "college_msg"
        elif configs[-1]["id"] == "ia_contacts_hypertext2009":
            configs[-1]["id"] = "ia_contacts_hypertext_2009"

        configs[-1]["id"] = configs[-1]["id"].lower()

for config in configs:
    Data = get_dataset_by_id(config["id"])
    data = Data("../../../data")

    es = data.get_event_sequence(shrink=True)

    dg = DynamicGraph()
    dg.init_from_event_sequence(
        event_sequence=es,
        time_resolution=data.resolution,
        directed=False,
        cumulative=data.cumulative,
    )

    dn2v = DynN2V(
        dim=config["dim"],
        p=config["p"],
        q=config["q"],
        num_walks=config["nw"],
        walk_length=config["wl"],
    )

    dataset_id = config["id"]
    dn2v.make_embedding(dg)
    dn2v.save_embedding(f"embs_dn2v/{dataset_id}")
