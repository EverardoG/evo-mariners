#!/bin/bash
#SBATCH -A kt-lab
#SBATCH --partition=preempt
#SBATCH -c 12
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --requeue
#SBATCH --nodelist=cn-v-[1-8]

cd ~/evo-mariners/apptainer
apptainer build --force --fakeroot ubuntu_20.04_ivp_2680_learn.sif docker://everardog/ubuntu_20.04_ivp_2680_learn:latest
