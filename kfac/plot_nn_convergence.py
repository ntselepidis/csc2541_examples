from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import os
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from parse import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='runs_curves', type=str)
    parser.add_argument('--scalar', default='training_error',
            choices=[
                'batch_size',
                'alpha',
                'beta',
                'gamma',
                'lambda',
                'quadratic_decrease',
                'training_error',
                'training_error_avg',
                'training_objective',
                'training_objective_avg',
                'test_error',
                'test_error_avg',
                'test_objective',
                'test_objective_avg'],
            type=str)
    parser.add_argument('--nbasis', default=1, type=int)
    parser.add_argument('--momentum', default=2, choices=[0, 1, 2], type=int)
    parser.add_argument('--iter_stop', default=-1, type=int)
    parser.add_argument('--logscale', default=1, choices=[0, 1], type=int)
    parser.add_argument('--output', default='pdf', choices=['pdf', 'png'], type=str)
    parser.add_argument('--dpi', default=300, type=int)
    parser.add_argument('--keep', default=None, type=str) # keep matches
    parser.add_argument('--drop', default=None, type=str) # drop matches
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()

    sns.set_style("darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111)

    df_list = []

    experiment = None

    for root, _, files in os.walk(args.logdir):
        if files == []:
            continue

        if 'cg_benchmark' in root:
            print(f'Skipping {root}/{files[0]} ...')
            continue
        else:
            print(f'Processing {root}/{files[0]} ...')

        _date, _time, _node, experiment, optimizer, nbasis, momentum, init_lambda, adapt_gamma, comment = parse(
                "{}_{}_{}_{}_{}_nbasis-{}_mom-{}_init-lambda-{}_adapt-gamma-{}_{}", os.path.basename(os.path.normpath(root)))

        seed = int(comment[-1])

        if (args.keep != None) and (args.keep not in root):
            continue
        if (args.drop != None) and (args.drop in root):
            continue

        if args.scalar == 'beta' and int(momentum) == 0:
            continue
        if args.momentum != 2 and args.momentum != int(momentum):
            continue
        if 'cgc' in optimizer:
            if int(nbasis) == args.nbasis:
                optimizer = optimizer + '(' + nbasis + ')'
            else:
                continue
        if int(momentum) == 1:
            optimizer = optimizer + ' with mom'

        event_acc = EventAccumulator(root)
        event_acc.Reload()
        times, iters, values = zip(*event_acc.Scalars('data/' + args.scalar))

        ndata = len(iters)

        data = {'iter': list(iters), args.scalar: list(values), 'optimizer': ndata * [optimizer], 'seed': ndata * [seed] }

        df_list.append(pd.DataFrame(data=data))

    df = pd.concat(df_list, ignore_index=True)

    if args.iter_stop > 0:
        df = df[df['iter'] < args.iter_stop]

    # create categorical optimizers for easier sorting
    df['optimizer_cat'] = pd.Categorical(
            df['optimizer'],
            categories=[
                'kfac',
                'kfac-cgc(' + str(args.nbasis) + ')',
                'kfac-cgc-m3(' + str(args.nbasis) + ')',
                'kfac with mom',
                'kfac-cgc(' + str(args.nbasis) + ') with mom',
                'kfac-cgc-m3(' + str(args.nbasis) + ') with mom',
                'conjgrad',
                'kfac-conjgrad',
                'kfac-cgc-conjgrad(' + str(args.nbasis) + ')',
                'kfac-cgc-m3-conjgrad(' + str(args.nbasis) + ')',
                ],
            ordered=True,
            )
    df = df.sort_values('optimizer_cat', kind='stable')

    color = sns.color_palette()

    palette = {
            'kfac'                                           : color[1],
            'kfac-cgc(' + str(args.nbasis) + ')'             : color[2],
            'kfac-cgc-m3(' + str(args.nbasis) + ')'          : color[3],
            'kfac with mom'                                  : color[1],
            'kfac-cgc(' + str(args.nbasis) + ') with mom'    : color[2],
            'kfac-cgc-m3(' + str(args.nbasis) + ') with mom' : color[3],
            'conjgrad'                                       : color[0],
            'kfac-conjgrad'                                  : color[1],
            'kfac-cgc-conjgrad(' + str(args.nbasis) + ')'    : color[2],
            'kfac-cgc-m3-conjgrad(' + str(args.nbasis) + ')' : color[3]
            }

    solid = ()
    dotted = (1, 1)
    dashed = (2, 2)

    dashes = {
            'kfac'                                           : dotted,
            'kfac-cgc(' + str(args.nbasis) + ')'             : dotted,
            'kfac-cgc-m3(' + str(args.nbasis) + ')'          : dotted,
            'kfac with mom'                                  : solid,
            'kfac-cgc(' + str(args.nbasis) + ') with mom'    : solid,
            'kfac-cgc-m3(' + str(args.nbasis) + ') with mom' : solid,
            'conjgrad'                                       : dashed,
            'kfac-conjgrad'                                  : dashed,
            'kfac-cgc-conjgrad(' + str(args.nbasis) + ')'    : dashed,
            'kfac-cgc-m3-conjgrad(' + str(args.nbasis) + ')' : dashed
            }

    sns.lineplot(data=df, x='iter', y=args.scalar, hue='optimizer', palette=palette, style='optimizer', dashes=dashes)

    plt.title(f'{experiment.upper()}: {args.scalar.replace("_", " ").title()} vs Iterations')
    plt.ylabel(f'{args.scalar.replace("_", " ").title()}')
    plt.xlabel('Iterations')

    if args.logscale:
        ax.set_yscale('log')

    output_dir = f'nn_convergence_plots_{os.path.basename(os.path.normpath(args.logdir))}'
    output_file = f'{args.scalar}_mom-{args.momentum}_nbasis-{args.nbasis}_iter-stop-{args.iter_stop}_logscale-{args.logscale}'
    if args.keep != None:
        output_file += f'_keep-{args.keep}'
    if args.drop != None:
        output_file += f'_drop-{args.drop}'
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f'{output_dir}/{output_file}.{args.output}', dpi=args.dpi)
