#!/bin/bash

declare -a scalars=("batch_size" \
                    "alpha" \
                    "beta" \
                    "gamma" \
                    "lambda" \
                    "quadratic_decrease" \
                    "training_error" \
                    "training_objective" \
                    "test_error" \
                    "test_objective")

#group=es_math

for experiment in "curves" "mnist"; do
    for scalar in ${scalars[@]}; do
        for nbasis in 1 5 10 20; do
            # plots for single step runs with and without momentum
            bsub -W 01:00 -n 2 -R "rusage[mem=8192]" \
                -oo ${SCRATCH}/kfac/nn_convergence_${experiment}_${scalar}_${nbasis}_drop-conjgrad.txt \
                python plot_nn_convergence.py --logdir ${SCRATCH}/kfac/results/${experiment} \
                                              --scalar ${scalar} \
                                              --iter_stop 2500 \
                                              --nbasis $nbasis \
                                              --drop conjgrad

            if [ ${scalar} != "beta" ]; then
                # plots for runs with conjgrad and varying `conjgrad_maxiters` (without momentum)
                for conjgrad_maxiter in 5; do
                    bsub -W 01:00 -n 2 -R "rusage[mem=8192]" \
                        -oo ${SCRATCH}/kfac/nn_convergence_${experiment}_${scalar}_${nbasis}_keep-maxiter-${conjgrad_maxiter}.txt \
                        python plot_nn_convergence.py --logdir ${SCRATCH}/kfac/results/${experiment} \
                                                      --scalar ${scalar} \
                                                      --iter_stop 2500 \
                                                      --nbasis $nbasis \
                                                      --keep maxiter-${conjgrad_maxiter}
                done # conjgrad_maxiter
            fi
        done # nbasis
    done # scalar
done # experiment
