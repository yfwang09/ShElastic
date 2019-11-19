#!/bin/sh

#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=ShElastic
#################
# time you think you need; default is one hour
#SBATCH --time=14-00:00:00
#################
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=24

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

# INSERT COMMANDS HERE

wait
