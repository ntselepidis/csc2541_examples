#!/bin/bash

for experiment in "curves" "mnist"; do
    for optimizer in "kfac" "kfac_cgc" "kfac_woodbury_v1" "kfac_woodbury_v2"; do
        bsub -G es_math -W 04:00 -n 4 -R "rusage[mem=8192]" -oo ${SCRATCH}/kfac/${experiment}_${optimizer}.txt python ${experiment}.py --optimizer ${optimizer}
    done
done
