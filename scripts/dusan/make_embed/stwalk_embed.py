from tigra.data.loaders import get_dataset_by_id
from tigra.core.dgraph import DynamicGraph
from tigra.embeddings.sota.stwalk import STWalk

import os

if not os.path.isdir("embs_stwalk"):
    os.mkdir("embs_stwalk")

configs = []

with open("../../../stwalk.conf", "r", encoding="utf-8") as f:
    f.readline()  # heading
    for line in f.readlines():
        config = line.split(",")
        configs.append(
            {
                "dataset": config[0],
                "id": config[0].replace("-", "_"),
                "dim": int(config[1]),
                "time_window": int(config[2]),
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

    dg = DynamicGraph()
    dg.init_from_event_sequence(
        event_sequence=es,
        time_resolution=data.resolution,
        directed=False,
        cumulative=data.cumulative,
    )

    stwalk = STWalk(
        dimension=config["dim"],
        time_window_size=config["time_window"],
        number_restart=10,
        walk_length=32,
        time_step=len(dg.get_graph_sequence()),
        workers=16,
        vocab_window_size=5,
    )

    dataset_id = config["id"]
    stwalk.make_embedding(dg)
    stwalk.save_embedding(f"embs_stwalk/{dataset_id}")
