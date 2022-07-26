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
    parser.add_argument('--filename', default='iter-249.csv', type=str)
    parser.add_argument('--logscale', default=0, choices=[0, 1], type=int)
    parser.add_argument('--stop_iter', default=-1, type=int)
    parser.add_argument('--output', default='png', choices=['pdf', 'png'])
    parser.add_argument('--dpi', default=300)
    parser.add_argument('--force_format', default=0, choices=[0, 1], type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    print(f'Reading {args.filename} ...')
    df = pd.read_csv(args.filename)

    experiment, nbasis, iteration = parse(
            'cg_benchmarks/cg_benchmark_{}_nbasis-{}_default_GPU_cg_benchmark_0/iter-{}.csv', os.path.normpath(args.filename))

    experiment, nbasis, iteration = experiment.upper(), int(nbasis), int(iteration) + 1

    if args.stop_iter > 0:
        df = df[df['iter'] <= args.stop_iter]

    #df = df[~(df['prec'].str.contains('m1|m2|kfac-m3|Qb'))]
    df = df[~(df['prec'].str.contains('x0'))]

    drop_precs = [
        #'none',
        #'kfac',
        #'kfac-cgc',
        'kfac-cgc-m1',
        'kfac-cgc-m2',
        #'kfac-cgc-m3',
        'kfac-m3',
        'kfac-m2',
        'kfac-cgc-m1-Qb',
        'kfac-cgc-m2-Qb',
        'kfac-cgc-m3-Qb',
        'kfac-m3-Qb',
        'kfac-m2-Qb',
        'none-x0',
        'kfac-x0',
        'kfac-cgc-x0',
        'kfac-cgc-m1-x0',
        'kfac-cgc-m2-x0',
        'kfac-cgc-m3-x0',
        'kfac-m3-x0',
        'kfac-m2-x0',
        'kfac-cgc-m1-Qb-plus-Ptx0',
        'kfac-cgc-m2-Qb-plus-Ptx0',
        'kfac-cgc-m3-Qb-plus-Ptx0',
        'kfac-m3-Qb-plus-Ptx0',
        'kfac-m2-Qb-plus-Ptx0']

    for drop_prec in drop_precs:
        df = df[~(df['prec'] == drop_prec)]

    sns.set_style("darkgrid")

    df.loc[ (df['prec'] != 'none') & (df['prec'] != 'kfac'), 'prec' ] += '(' + str(nbasis) + ')'

    # Attention !!!
    # The code in the branch below is optional and needs to be updated whenever
    # drop_precs list is modified. This is only a hacky way for ensuring color
    # consistency of the methods/lines in my thesis.
    palette = None
    if args.force_format:
        # create categorical optimizers for easier sorting
        df['prec_cat'] = pd.Categorical(
                df['prec'],
                categories=[
                    'none',
                    'kfac',
                    'kfac-cgc(' + str(nbasis) + ')',
                    'kfac-cgc-m1(' + str(nbasis) + ')',
                    'kfac-cgc-m2-Qb(' + str(nbasis) + ')',
                    'kfac-cgc-m3(' + str(nbasis) + ')',
                    'kfac-m3-Qb(' + str(nbasis) + ')',
                    'kfac-m2-Qb(' + str(nbasis) + ')'
                    ],
                ordered=True,
                )
        df = df.sort_values('prec_cat', kind='stable')

        color = sns.color_palette()
        palette = {
                'none' : color[0],
                'kfac' : color[1],
                'kfac-cgc(' + str(nbasis) + ')' : color[2],
                'kfac-cgc-m1(' + str(nbasis) + ')': color[4],
                'kfac-cgc-m2-Qb(' + str(nbasis) + ')': color[5],
                'kfac-cgc-m3(' + str(nbasis) + ')': color[3],
                'kfac-m3-Qb(' + str(nbasis) + ')': color[6],
                'kfac-m2-Qb(' + str(nbasis) + ')': color[7],
                }

    fig, axs = plt.subplots(1, 2, figsize=(2*6.4, 4.8))
    #fig.suptitle(f'conjgrad convergence plots ( {args.filename} )')
    fig.suptitle(f'{experiment}: Convergence of CG at step {iteration} of non-linear optimization')

    sns.lineplot(ax=axs[0], data=df, x="iter", y="val", hue="prec", palette=palette)
    sns.lineplot(ax=axs[1], data=df, x="iter", y="relres", hue="prec", palette=palette)

    axs[0].set(xlabel='CG Iterations', ylabel='Value of Quadratic')
    axs[1].set(xlabel='CG Iterations', ylabel='Relative Residual')

    # change title of legends ('prec' => 'preconditioner')
    for i in range(2):
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].legend(handles=handles, labels=labels, title="preconditioner")

    if args.logscale:
        axs[1].set_yscale('log')

    plt.savefig(f'{args.filename[0:-4]}.{args.output}', dpi=args.dpi)

if __name__ == '__main__':
    main()
