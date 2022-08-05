#!/bin/bash

dir=$1

stop_iter=150
logscale=0
comment="default"
force_format=1

for f in `ls -v $dir/*.csv`; do
    echo "Generating plot for $f ..."
    python plot_cg_convergence.py --filename $f --logscale $logscale --stop_iter $stop_iter --comment $comment --force_format $force_format
done
