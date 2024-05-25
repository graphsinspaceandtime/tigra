#!/bin/bash
#SBATCH --job-name=tigra
#SBATCH --partition=cuda
#SBATCH --cpus-per-task=<cores>
#SBATCH --mem=<memory>
#SBATCH --nodelist=<target node>
#SBATCH --output slurm.%J.out
#SBATCH --error slurm.%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your email>

### Usage
### Position yourself in directory where you want to store log files
### (slurm.id.out and slurm.id.err)
###
### $ sbatch <path to sbatch>/slurm_template.sbatch
###
### Example fs structure:
### /home/user
### |-tigra
###   |-slurm_template.sbatch
### |-logs
###   |-*current location*
###
### Run the sbatch script from logs directory so all logs are stored in that
### directory.
###
### $ sbatch ../tigra/slurm_template.sbatch

### Module init
. /usr/share/modules/init/bash

### Load environment
module purge
module load python/miniconda3

### user commands
eval "$(conda shell.bash hook)" # required for conda
conda activate tigra            # activate conda env
cd tests                        # go to tests directory
python test_databank.py         # run a script
conda deactivate                # deactivate conda env
