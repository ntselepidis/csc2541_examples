#!/bin/bash

rootdir=$1

stop_iter=150
logscale=0

for dir in `ls $rootdir`; do
    echo $dir
    for f in `ls $rootdir/$dir`; do
        echo "Generating plot for $rootdir/$dir/$f ..."
        python plot_cg_convergence.py --filename $rootdir/$dir/$f --logscale $logscale --stop_iter $stop_iter
    done
done
