#!/bin/bash

#SBATCH --partition=math
#SBATCH --nodes=1
#SBATCH --ntasks=21
#SBATCH --mem-per-cpu=1400MB
#SBATCH --gres=gpu:1

for filename in ./*.slurm; do
        sbatch "$filename"
done
