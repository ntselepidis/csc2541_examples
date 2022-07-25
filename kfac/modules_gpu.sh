#!/bin/bash

echo "Switching to new software stack"
env2lmod

echo "Switching to GCC 8.2.0"
module load gcc/8.2.0

echo "Loading base modules"
module load bash/5.0
module load git/2.31.1
module load vim/8.2.1201

#module load python/3.9.9
module load python_gpu/3.9.9
#module load cuda/11.4.2

source ~/venv-gpu/bin/activate
