from tigra.data.loaders import get_dataset_by_id
from tigra.core.dgraph import DynamicAggregateGraph
from tigra.embeddings.sota.ctdne_wrapper import Ctdne

import os

if not os.path.isdir("embs_ctdne"):
    os.mkdir("embs_ctdne")

configs = []

with open("../../../ctdne.conf", "r", encoding="utf-8") as f:
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
            }
        )
        if configs[-1]["id"] == "CollegeMsg":
            configs[-1]["id"] = "college_msg"
        elif configs[-1]["id"] == "ia_contacts_hypertext2009":
            configs[-1]["id"] = "ia_contacts_hypertext_2009"

        configs[-1]["id"] = configs[-1]["id"].lower()

for config in configs:
    print(config["id"])
    Data = get_dataset_by_id(config["id"])
    data = Data("../../../data")

    es = data.get_event_sequence(shrink=True)

    dg = DynamicAggregateGraph(
        name="",
        directed=False,
        event_sequence=es,
        weighted=False,
    )

    stwalk = Ctdne(
        dim=config["dim"],
        p=config["p"],
        q=config["q"],
    )

    dataset_id = config["id"]
    stwalk.make_embedding(dg)
    stwalk.save_embedding(f"embs_ctdne/{dataset_id}")
