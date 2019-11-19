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

python3 script_shape1_b1e+00_g1e-02.py > script_shape1_b1e+00_g1e-02.txt &
python3 script_shape1_b1e+00_g3e-02.py > script_shape1_b1e+00_g3e-02.txt &
python3 script_shape1_b1e+00_g1e-01.py > script_shape1_b1e+00_g1e-01.txt &
python3 script_shape1_b1e+00_g3e-01.py > script_shape1_b1e+00_g3e-01.txt &
python3 script_shape1_b1e+00_g1e+00.py > script_shape1_b1e+00_g1e+00.txt &
python3 script_shape1_b1e+00_g3e+00.py > script_shape1_b1e+00_g3e+00.txt &
python3 script_shape1_b1e+00_g1e+01.py > script_shape1_b1e+00_g1e+01.txt &
python3 script_shape1_b1e+00_g3e+01.py > script_shape1_b1e+00_g3e+01.txt 
python3 script_shape2_b1e+00_g1e-02.py > script_shape2_b1e+00_g1e-02.txt &
python3 script_shape2_b1e+00_g3e-02.py > script_shape2_b1e+00_g3e-02.txt &
python3 script_shape2_b1e+00_g1e-01.py > script_shape2_b1e+00_g1e-01.txt &
python3 script_shape2_b1e+00_g3e-01.py > script_shape2_b1e+00_g3e-01.txt &
python3 script_shape2_b1e+00_g1e+00.py > script_shape2_b1e+00_g1e+00.txt &
python3 script_shape2_b1e+00_g3e+00.py > script_shape2_b1e+00_g3e+00.txt &
python3 script_shape2_b1e+00_g1e+01.py > script_shape2_b1e+00_g1e+01.txt &
python3 script_shape2_b1e+00_g3e+01.py > script_shape2_b1e+00_g3e+01.txt
python3 script_shape3_b1e+00_g1e-02.py > script_shape3_b1e+00_g1e-02.txt &
python3 script_shape3_b1e+00_g3e-02.py > script_shape3_b1e+00_g3e-02.txt &
python3 script_shape3_b1e+00_g1e-01.py > script_shape3_b1e+00_g1e-01.txt &
python3 script_shape3_b1e+00_g3e-01.py > script_shape3_b1e+00_g3e-01.txt &
python3 script_shape3_b1e+00_g1e+00.py > script_shape3_b1e+00_g1e+00.txt &
python3 script_shape3_b1e+00_g3e+00.py > script_shape3_b1e+00_g3e+00.txt &
python3 script_shape3_b1e+00_g1e+01.py > script_shape3_b1e+00_g1e+01.txt &
python3 script_shape3_b1e+00_g3e+01.py > script_shape3_b1e+00_g3e+01.txt

wait
