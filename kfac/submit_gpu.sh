#!/bin/bash
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

comment="default_GPU"
group=ls_math
for experiment in "curves" "mnist"; do
    for optimizer in "kfac" "kfac-cgc" "kfac-woodbury-v1" "kfac-woodbury-v2"; do
        for random_seed in 0 1 2 3 4; do
            bsub -G ${group} -W 04:00 -n 4 -R "rusage[mem=8192,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" \
                -oo ${SCRATCH}/kfac/${experiment}_${optimizer}_${comment}_${random_seed}.txt \
                python ${experiment}.py --optimizer ${optimizer} --comment ${comment} --random_seed ${random_seed}
        done
    done
done
