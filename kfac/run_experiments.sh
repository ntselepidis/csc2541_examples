#!/bin/bash
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

comment="default_GPU"
group=ls_math
random_seed=0
use_momentum=1
init_lambda=150
adapt_gamma=0

for experiment in "curves" "mnist"; do

    # CG benchmark
    for random_seed in 0; do
        for nbasis in 1 5 10 20; do
            bsub -G ${group} -W 04:00 -n 4 -R "rusage[mem=8192,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" \
                -oo ${SCRATCH}/kfac/${experiment}_kfac_${nbasis}_${comment}_${random_seed}_1_${init_lambda}_${adapt_gamma}_cg_benchmark.txt \
                python ${experiment}.py --optimizer kfac \
                                        --comment ${comment}_cg_benchmark \
                                        --random_seed ${random_seed} \
                                        --use_momentum 1 \
                                        --init_lambda ${init_lambda} \
                                        --adapt_gamma ${adapt_gamma} \
                                        --nbasis ${nbasis} \
                                        --conjgrad_maxiter 150 \
                                        --conjgrad_benchmark_interval 50
        done
    done

    # individual runs
    for random_seed in 0 1 2 3 4; do

        # one-level kfac run
        for optimizer in "kfac"; do
            for use_momentum in 1 0; do
                bsub -G ${group} -W 04:00 -n 4 -R "rusage[mem=8192,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" \
                    -oo ${SCRATCH}/kfac/${experiment}_${optimizer}_${comment}_${random_seed}_${use_momentum}_${init_lambda}_${adapt_gamma}.txt \
                    python ${experiment}.py --optimizer ${optimizer} \
                                            --comment ${comment} \
                                            --random_seed ${random_seed} \
                                            --use_momentum ${use_momentum} \
                                            --init_lambda ${init_lambda} \
                                            --adapt_gamma ${adapt_gamma}
            done
        done

        # unpreconditioned conjgrad and one-level kfac-conjgrad runs
        for optimizer in "conjgrad" "kfac-conjgrad"; do
            for conjgrad_maxiter in 5; do
                bsub -G ${group} -W 04:00 -n 4 -R "rusage[mem=8192,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" \
                    -oo ${SCRATCH}/kfac/${experiment}_${optimizer}_maxiter-${conjgrad_maxiter}_${comment}_${random_seed}_0_${init_lambda}_${adapt_gamma}.txt \
                    python ${experiment}.py --optimizer ${optimizer} \
                                            --comment maxiter-${conjgrad_maxiter}_${comment} \
                                            --random_seed ${random_seed} \
                                            --use_momentum 0 \
                                            --init_lambda ${init_lambda} \
                                            --adapt_gamma ${adapt_gamma} \
                                            --conjgrad_maxiter ${conjgrad_maxiter}
            done
        done

        for nbasis in 1 5 10 20; do

            # two-level kfac runs
            for optimizer in "kfac-cgc" "kfac-cgc-m3" "kfac-woodbury-v2"; do
                for use_momentum in 1 0; do
                    bsub -G ${group} -W 04:00 -n 4 -R "rusage[mem=8192,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" \
                        -oo ${SCRATCH}/kfac/${experiment}_${optimizer}_${nbasis}_${comment}_${random_seed}_${use_momentum}_${init_lambda}_${adapt_gamma}.txt \
                        python ${experiment}.py --optimizer ${optimizer} \
                                                --comment ${comment} \
                                                --random_seed ${random_seed} \
                                                --use_momentum ${use_momentum} \
                                                --init_lambda ${init_lambda} \
                                                --adapt_gamma ${adapt_gamma} \
                                                --nbasis ${nbasis}
                done
            done

            # two-level kfac-conjgrad runs
            for optimizer in "kfac-cgc-conjgrad" "kfac-cgc-m3-conjgrad" "kfac-woodbury-conjgrad-v2"; do
                for conjgrad_maxiter in 5; do
                    bsub -G ${group} -W 04:00 -n 4 -R "rusage[mem=8192,ngpus_excl_p=1]" -R "select[gpu_model0==NVIDIAGeForceGTX1080Ti]" \
                        -oo ${SCRATCH}/kfac/${experiment}_${optimizer}_maxiter-${conjgrad_maxiter}_${nbasis}_${comment}_${random_seed}_0_${init_lambda}_${adapt_gamma}.txt \
                        python ${experiment}.py --optimizer ${optimizer} \
                                                --comment maxiter-${conjgrad_maxiter}_${comment} \
                                                --random_seed ${random_seed} \
                                                --use_momentum 0 \
                                                --init_lambda ${init_lambda} \
                                                --adapt_gamma ${adapt_gamma} \
                                                --nbasis ${nbasis} \
                                                --conjgrad_maxiter ${conjgrad_maxiter}
                done
            done

        done # nbasis

    done # random_seed

done # experiment
