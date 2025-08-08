#!/bin/bash
#SBATCH -A kt-lab
#SBATCH --partition=share
#SBATCH -c 12
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00

cd ~/evo-mariners/apptainer
./build_image.sh
