# tigra

TIGRA

Graphs in Space and Time: Graph Embeddings for Machine Learning in Complex Dynamical Systems

Tigra software library

TODO:
- autori
- uputstva za condu
- tutorijali
- etc.

## Set up Conda environment

### Install Conda

Only Linux distributions are documented here since they are a bit difficult to
handle. Windows and macOS should be straight forward if you follow the Conda
[installation documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

#### Ubuntu

```sh
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash
```

### Fedora

```sh
sudo dnf install conda
```

# Configure Conda environment

Initialize Conda for shell interaction. This is requred before doing any of the
following steps. It is recommended to restart the shell session (terminal
application) before moving on to the next step.

```sh
conda init
```

Create the preconfigured environment for the TIGRA library. It contains most of
the required libraries for testing and usage of all TIGRA software.

```sh
conda env create -f environment.yaml
```

If environment configuration sometimes changes, the created environment should
be updated accordingly. The following command does that. It is not required to
execute the command all the time, only when the yaml file changes.

```sh
conda env update -f environment.yaml
```

Activate the conda environment. This configures shell parameters to use packages
and libraries that are installed in the environment.

```sh
conda activate tigra
```

Install remaining packages which are not available in PyPl repositores. Do this
only once you're in the conda environment.

```sh
pip install -r requirements.txt
```

When you're done working with the software, you can exit the environment by
deactivating it.

```sh
conda deactivate
```

### Conda FAQ

1. Q: There is `base` in the shell (terminal) prompt after I install conda.

   A: This is the base environment in conda which is automatically activated.
      Automatic activation can be disabled. https://stackoverflow.com/a/57642532
