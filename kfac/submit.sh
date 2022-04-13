#!/bin/bash

comment="default"
for experiment in "curves" "mnist"; do
    for optimizer in "kfac" "kfac_cgc" "kfac_woodbury_v1" "kfac_woodbury_v2"; do
        for random_seed in 0 1 2 3 4; do
            bsub -G es_math -W 04:00 -n 4 -R "rusage[mem=8192]" -oo ${SCRATCH}/kfac/${experiment}_${optimizer}_${comment}_${random_seed}.txt \
                python ${experiment}.py --optimizer ${optimizer} --comment ${comment} --random_seed ${random_seed}
        done
    done
done
