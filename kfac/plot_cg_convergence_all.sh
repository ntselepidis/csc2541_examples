#!/bin/bash

rootdir=$1

for dir in `ls -v $rootdir`; do
    bsub -W 01:00 -n 2 -R "rusage[mem=8192]" \
        -oo ${SCRATCH}/kfac/cg_convergence_${dir}.txt \
        ./plot_cg_convergence.sh ${rootdir}/${dir}
done
