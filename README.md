# TIGRA

> Graphs in Space and Time : Graph Embeddings for Machine Learning in Complex
> Dynamical Systems

Tigra software library

TODO:
- autori
- uputstva za condu
- tutorijali
- etc.

## Downloading datasets

The datasets are not stored inside the git repository since they are quite
large. They can however be downloaded using a simple bash script
[get_datasets.sh](data/get_datasets.sh). You can run it using the `bash`
command.

```sh
cd data
bash get_datasets.sh
```

This scripts depends on `curl` command so make sure it's installed on your
system.

## Set up Conda environment

### Install Conda

Only Linux distributions are documented here since they are a bit difficult to
handle. Windows and macOS should be straight forward if you follow the Conda
[installation documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

#### Ubuntu

```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
echo 'export PATH="$PATH:$HOME/miniconda3/condabin"' | tee -a $HOME/.bash_profile
```

#### Fedora

```sh
sudo dnf install conda
```

### Configure Conda environment

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

There is also an option to add machine specific configuration of the
environment such as some variables. This will allows easier maintanance and
configuration between machines with different hardware specifications. In this
repository it is a convention to use the `local.yaml` configuration file for
that. An example of such file is available in [local.example.yaml](./local.example.yaml)
file. It can be applied by updating the created environment.

```sh
conda env update -f local.yaml
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

Also needed for developent (for using tigra as a global package inside the
module itself) you will need to install the module as an editable with pip. To
do this, run the following command in the root of the repository.

```sh
python setup.py develop
```

When you're done working with the software, you can exit the environment by
deactivating it.

```sh
conda deactivate
```

### Other environments

There are some other conda environments which are used for libraries that are
hard to configure with other environments. To activate them, run the same
commands for creating, updating, activating and deactivating as for the main
environment but with a different file specified.

| Environment                | File                                                                                   |
|----------------------------|----------------------------------------------------------------------------------------|
| PyTorch Geometric Temporal | [environment.torch_geometric_temporal.yaml](environment.torch_geometric_temporal.yaml) |

### Conda FAQ

1. Q: There is `base` in the shell (terminal) prompt after I install conda.

   A: This is the base environment in conda which is automatically activated.
      Automatic activation can be disabled. https://stackoverflow.com/a/57642532
