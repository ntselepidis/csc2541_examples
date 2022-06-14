#!/bin/bash

dir=$1

stop_iter=100
logscale=0

for f in `ls $dir`; do
    echo "Generating plot for $dir/$f ..."
    python plot_cg_convergence.py --filename $dir/$f --logscale $logscale --stop_iter $stop_iter
done
