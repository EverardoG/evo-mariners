#!/bin/bash
#SBATCH -A kt-lab
#SBATCH --partition=preempt
#SBATCH -c 12
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=cn-v-[1-9]

which apptainer
ldd "$(which apptainer)" | grep libc.so

source ~/hpc-share/miniforge/bin/activate
conda activate mariners

cd ~/evo-mariners/
python simple_evo.py /nfs/stak/users/gonzaeve/hpc-share/2025-08-06/kelpbass
