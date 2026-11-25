<p align="center">
  <img src="https://tigraphs.pmf.uns.ac.rs/wp-content/uploads/2023/12/cropped-tigar-logo-final-3.png" alt="TIGRA logo" width="140">
</p>

<h1 align="center">TIGRA</h1>

<p align="center">
  <b>Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12">
  <a href="https://tigraphs.net/"><img src="https://img.shields.io/badge/project-website-green.svg" alt="Project website"></a>
</p>

TIGRA is a Python library for **learning, evaluating and applying node embeddings
of dynamic (temporal) graphs**. It turns a stream of timestamped interactions
into a sequence of graph snapshots, embeds each snapshot with one of many
dynamic embedding methods, and evaluates the embeddings on downstream tasks
such as link prediction, node classification, clustering, diffusion prediction
and anomaly detection. Project website: https://tigraphs.net/



## Features

**Dynamic graph core** (`tigra.core`)
- `EventSequence` / `Event`: timestamped interaction streams
- `DynamicGraph`: snapshot sequences built from events at a chosen time
  resolution (day, week, month, 3/6 months, year), optionally cumulative
- Graph reconstruction evaluation

**Embedding methods** (`tigra.embeddings`)

| Family | Methods |
|---|---|
| Random-walk baselines | `DynDeepWalk`, `DynN2V` (dynamic node2vec) |
| Hubness-aware walks | `HubRandomWalk`, `DynDeepHubWalk` |
| LID-aware walks | `DynN2V_LIDRW`, `DynN2V_LIDDelta`, `DynN2V_NWLID`, `DynN2V_PQLID`, `DynDeepLIDWalk` |
| Reinforcement learning | `DynRLWalk` (multi-armed bandit walk sampler) |
| Graph autoencoders | `BasicDynAE`, `HubDynAE`, `LIDDynAE`, `DistanceDynAE`, `HubDistanceDynAE`, `LIDDistanceDynAE` |
| State-of-the-art wrappers | CTDNE, dynnode2vec, STWalk, DynGEM, DynAE, DynRNN, DynAERNN, TIMERS (via CTGCN) |

Hyperparameter tuning helpers are in `tigra.embeddings.hypertuner` and
`tigra.embeddings.de_hypertuner`.

**Evaluation and applications** (`tigra.evaluation`)
- Link prediction (dot product, cosine, L1/L2, Hadamard, concatenation features)
- Node classification (scikit-learn, boosting, PyTorch MLP, TabPFN back-ends)
- Clustering
- Diffusion (SI process) prediction
- Anomaly detection
- Hubness and LID analysis of embedding spaces

**Synthetic generators** (`tigra.generators`)
- Dynamic Barabási-Albert (DBA), dynamic Erdős-Rényi (DER), (dynamic) stochastic
  block models (SBM, DSBM, SBTM)

---

## Installation

TIGRA is developed and tested on **Linux with Python 3.12** and uses
[Conda](https://docs.conda.io/) for dependency management.

```sh
git clone https://github.com/graphsinspaceandtime/tigra.git
cd tigra

conda env create -f environment.tigranew.yml
conda activate tigranew

pip install -r requirements.txt
pip install -e .
```

To update the environment after `environment.tigranew.yml` changes:

```sh
conda env update -f environment.tigranew.yml
```

<details>
<summary><b>Machine-specific settings (optional)</b></summary>

Machine-specific environment variables (e.g. the number of CPUs) can be kept in
a local, git-ignored `local.yaml`. Copy the example and apply it:

```sh
cp local.example.yaml local.yaml   # edit as needed
conda env update -f local.yaml
```
</details>

<details>
<summary><b>Other environments</b></summary>

Some libraries are hard to install together, so separate environments are
provided:

| Environment | File | Notes |
|---|---|---|
| Main | `environment.tigranew.yml` | Recommended |
| PyTorch Geometric Temporal | `environment.torch_geometric_temporal.yaml` | For PyG Temporal baselines |
| Windows | `win-environment.yaml` | Not regularly tested |

> **Note:** the `dynnode2vec` package only supports Python ≤ 3.8. Use a
> separate Python 3.8 environment if you need the `Dynnode2vec` wrapper.
</details>

<details>
<summary><b>Installing Conda on Linux</b></summary>

Ubuntu:

```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda init   # then restart the shell
```

Fedora:

```sh
sudo dnf install conda
conda init
```

To stop the `base` environment from activating automatically:
`conda config --set auto_activate_base false`.
</details>

---

## Datasets

Datasets are **not** stored in the repository. Download them with:

```sh
cd data
bash get_datasets.sh        # Linux / macOS (requires curl)
# or, on Windows:
# powershell -File get_datasets.ps1
```

This fetches the following temporal networks into `data/`:

| Source | Datasets |
|---|---|
| [Network Repository](https://networkrepository.com/) | `radoslaw-email`, `fb-forum`, `fb-messages`, `ia-contacts_hypertext2009`, `ia-enron-employees`, `ia-hospital-ward-proximity-attr`, `ia-primary-school-proximity-attr`, `ia-reality-call` |
| [SNAP](https://snap.stanford.edu/data/#temporal) | `CollegeMsg`, `email-Eu-core-temporal`, `sx-askubuntu`, `sx-mathoverflow`, `sx-superuser`, `wiki-talk-temporal` |
| Synthetic (included) | `synthetic-dba`, `synthetic-dbac`, `synthetic-der`, `synthetic-dpp` |


---

## Quick start

Run from the repository root, so that both `tigra` and the `data` package can be
imported:

```python
from tigra.core.loaders import load_event_sequence
from tigra.core.dgraph import DynamicGraph
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.rndw.rndwbase import DynN2V
from tigra.embeddings.rndw.rlwalk import DynRLWalk

# 1. Load a temporal network as a sequence of timestamped events
events, resolution = load_event_sequence("radoslaw-email")
print("events:", events.num_events(), "| days:", events.get_timespan_in_days())

# 2. Build a sequence of graph snapshots
dg = DynamicGraph()
dg.init_from_event_sequence(events, time_resolution=resolution, directed=False)
dg.info()

# 3. Embed every snapshot
model = DynRLWalk(dim=128, epsilon=0.1, alpha=0.1)  # or DynN2V(128, 1, 1), ...
model.make_embedding(dg)

# 4. Evaluate, e.g. graph reconstruction per snapshot
for t, graph in enumerate(dg.get_graph_sequence()):
    ev = GraphReconstructionEval(graph, model[t])
    print(f"snapshot {t}: F1 = {ev.getF1():.3f}")

# 5. Save embeddings (one .emb file per snapshot)
model.save_embedding("embeddings/radoslaw-email/DynRLWalk")
```

More complete examples are in `tests/` and `scripts/`.

---

## Repository structure

```
tigra/
├── core/          # Event sequences, dynamic graphs, loaders, reconstruction eval
├── data/          # Data loading utilities
├── embeddings/
│   ├── rndw/      # Random-walk methods (DeepWalk, node2vec, hubness, LID, RL, hybrid)
│   ├── ae/        # Dynamic graph autoencoders
│   └── sota/      # Wrappers around state-of-the-art methods
├── evaluation/    # Link prediction, classification, clustering, diffusion, anomalies, hubness, LID
└── generators/    # Synthetic dynamic graph generators
data/              # Dataset download scripts and the Databank loader
configs/           # Experiment / hyperparameter configurations
tests/             # Tests and example experiment scripts
scripts/           # Experiment scripts
```


---

## Authors


TIGRA is developed at the Faculty of Sciences, University of Novi Sad, Serbia.

- Miloš Radovanović
- Vladimir Kurbalija
- Miloš Stojaković
- Mirjana Ivanović
- Miloš Savić
- Marko Savić
- Dušica Knežević
- Aleksandar Tomčić
- Jela Babić
- Dušan Simić

Contributions are welcome. Please open an issue or a pull request.

---

## Acknowledgements

This work was supported by the Science Fund of the Republic of Serbia,
<a href="https://tigraphs.net/">project TIGRA</a> (*Graphs in Space and Time: Graph Embeddings for Machine Learning
in Complex Dynamical Systems*), grant no. #7462.


---

## Third-party code

TIGRA includes code adapted from the following MIT-licensed projects. Their
original license notices are kept in the corresponding directories:

| Component | Location | Upstream |
|---|---|---|
| CTGCN (and its DynGEM / DynAE / DynRNN / DynAERNN / TIMERS baselines) | `tigra/embeddings/sota/CTGCN/` | [jhljx/CTGCN](https://github.com/jhljx/CTGCN) |
| CTDNE | `tigra/embeddings/sota/ctdnesrc/` | [urielsinger/CTDNE](https://github.com/urielsinger/CTDNE) |
| STWalk | `tigra/embeddings/sota/stwalk.py` | [supriya-gdptl/STWalk](https://github.com/supriya-gdptl/STWalk) |

---

## License

TIGRA is released under the [Apache License 2.0](LICENSE). 

Copyright © 2023–2026 The TIGRA authors.
