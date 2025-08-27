#!/bin/bash
#SBATCH -A kt-lab
#SBATCH --partition=preempt
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=cn-v-[1-9]
#SBATCH --job-name=tar-influence-shaping

tar --use-compress-program="pigz -p 12" -vcf /nfs/stak/users/gonzaeve/hpc-share/influence-shaping.tar.gz /nfs/stak/users/gonzaeve/hpc-share/influence-shaping
