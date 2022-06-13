from jax import nn, numpy as np
from jax.example_libraries import stax
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
import time
import os

import kfac
import kfac_util


def get_architecture(input_size, layer_sizes):
    """Construct a sigmoid MLP autoencoder architecture with the given layer sizes.
    The code layer, given by the name 'code', is linear."""
    
    layers = []
    param_info = []
    act_name = 'in'
    for name, lsize in layer_sizes:
        if name == 'code':
            # Code layer is special because it's linear
            param_info.append((act_name, name))
            act_name = name

            layers.append((name, stax.Dense(
                lsize, W_init=kfac_util.sparse_init(), b_init=nn.initializers.zeros)))
        else:
            preact_name = name + 'z'
            param_info.append((act_name, preact_name))
            act_name = name + 'a'

            layers.append((preact_name, stax.Dense(
                lsize, W_init=kfac_util.sparse_init(), b_init=nn.initializers.zeros)))
            layers.append((act_name, stax.elementwise(nn.sigmoid)))

    layers.append(('out', stax.Dense(
        input_size, W_init=kfac_util.sparse_init(), b_init=nn.initializers.zeros)))

    param_info.append((act_name, 'out'))
    param_info = tuple(param_info)

    net_init, net_apply = kfac_util.named_serial(*layers)

    in_shape = (-1, input_size)
    flatten, unflatten = kfac_util.get_flatten_fns(net_init, in_shape)

    return kfac_util.Architecture(net_init, net_apply, in_shape, flatten, unflatten, param_info)


def default_config():
    config = {}
    config['max_iter'] = 20000
    config['initial_batch_size'] = 1000
    config['final_batch_size_iter'] = 500
    config['batch_size_granularity'] = 50
    config['chunk_size'] = 5000
    
    config['cov_update_interval'] = 1
    config['cov_batch_ratio'] = 1/8
    config['cov_timescale'] = 20
    
    config['eig_update_interval'] = 20

    config['lambda_update_interval'] = 5
    config['init_lambda'] = 150
    config['lambda_drop'] = 0.95
    config['lambda_boost'] = 1 / config['lambda_drop']
    config['lambda_min'] = 0
    config['lambda_max'] = onp.infty

    config['weight_cost'] = 1e-5

    config['gamma_update_interval'] = 20
    config['init_gamma'] = onp.sqrt(config['init_lambda'] + config['weight_cost'])
    config['gamma_drop'] = onp.sqrt(config['lambda_drop'])
    config['gamma_boost'] = 1 / config['gamma_drop']
    config['gamma_max'] = 1
    config['gamma_min'] = onp.sqrt(config['weight_cost'])
    
    config['param_timescale'] = 100

    config['comment'] = 'default'
    config['random_seed'] = 0
    config['use_momentum'] = 1
    config['adapt_gamma'] = 1
    config['conjgrad_tol'] = 5e-4
    config['conjgrad_maxiter'] = 100
    config['conjgrad_benchmark_interval'] = config['max_iter'] + 10 # disabled by default
    
    return config

def squared_error(logits, T):
    """Compute the squared error. For consistency with James's code, don't
    rescale by 0.5."""
    y = nn.sigmoid(logits)
    return np.sum((y-T)**2)

def plot_matrix_to_tensorboard(writer, optimizer, mat, comment, step):
    if 'woodbury' in optimizer:
        mat = mat - np.diag(np.diag(mat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat)
    fig.colorbar(cax)
    fig.canvas.draw()
    writer.add_figure(comment, fig, step)
    plt.close(fig)

# def plot_natgrad_to_tensorboard(writer, natgrad_w, natgrad_w_corr, comment, step):
#     fig, axs = plt.subplots(2)
#     fig.suptitle(comment)
#     axs[0].plot(natgrad_w)
#     axs[1].plot(natgrad_w_corr)
#     axs[0].set_yscale('log')
#     axs[1].set_yscale('log')
#     fig.canvas.draw()
#     writer.add_figure(comment, fig, step)
#     plt.close(fig)

def plot_conjgrad_convergence_to_tensorboard(writer, val, relres, comment, step):
    fig, axs = plt.subplots(1, 2, figsize=(2*6.4, 4.8))
    fig.suptitle('conjgrad convergence plots')

    for key in val:
        axs[0].plot(val[key], label=key)
    axs[0].set_title('val')
    axs[0].grid()

    for key in relres:
        axs[1].plot(relres[key], label=key)
    axs[1].set_title('relres')
    axs[1].set_yscale('log')
    axs[1].grid()

    plt.legend()

    fig.canvas.draw()
    writer.add_figure(comment, fig, step)
    plt.close(fig)

def get_conjgrad_convergence_dataframe(val, relres):
    keys = list(val.keys())
    niter = len(val[keys[0]])

    _prec = []
    _iter = []
    _val = []
    _relres = []
    for key in keys:
        _prec = _prec + (niter*[key])
        _iter = _iter + [i for i in range(niter)]
        _val = _val + val[key].tolist()
        _relres = _relres + relres[key].tolist()

    d = {'prec': _prec, 'iter': _iter, 'val': _val, 'relres': _relres}

    return pd.DataFrame(data=d)

def run_training(X_train, X_test, arch, config):
    if 'conjgrad' in config['optimizer']:
        config['use_momentum'] = 0
    writer = SummaryWriter(comment='_' + config['experiment'] + '_' + \
            config['optimizer'] + '_mom-' + str(config['use_momentum']) + \
            '_init-lambda-' + str(int(config['init_lambda'])) + \
            '_adapt-gamma-' + str(int(config['adapt_gamma'])) + '_' + \
            config['comment'] + '_' + str(config['random_seed']))
    nll_fn = kfac_util.BernoulliModel.nll_fn
    state = kfac.kfac_init(arch, kfac_util.BernoulliModel, X_train, X_train, config, config['random_seed'])
    for i in range(config['max_iter']):
        t0 = time.time()
        state = kfac.kfac_iter(state, arch, kfac_util.BernoulliModel, X_train, X_train, config)

        print('Step', i)
        print('Time:', time.time() - t0)
        print('Batch size:', state['batch_size'])
        writer.add_scalar('data/batch_size', state['batch_size'], i)
        print('Alpha:', state['coeffs'][0])
        writer.add_scalar('data/alpha', state['coeffs'][0], i)
        if config['use_momentum'] and i > 0:
            print('Beta:', state['coeffs'][1])
            writer.add_scalar('data/beta', state['coeffs'][1], i)
        print('Quadratic decrease:', state['quad_dec'])
        writer.add_scalar('data/quadratic_decrease', state['quad_dec'], i)

        if 'conjgrad' in config['optimizer']:
            print('CG niters:', state['conjgrad_niters'])
            writer.add_scalar('data/conjgrad_niters', state['conjgrad_niters'], i)

        if i % 20 == 0:
            print()
            cost = kfac.compute_cost(arch, nll_fn, state['w'], X_train, X_train, 
                config['weight_cost'], config['chunk_size'])
            print('Training objective:', cost)
            writer.add_scalar('data/training_objective', cost, i)
            cost = kfac.compute_cost(
                arch, nll_fn, state['w_avg'], X_train, X_train, 
                config['weight_cost'], config['chunk_size'])
            print('Training objective (averaged):', cost)
            writer.add_scalar('data/training_objective_avg', cost, i)

            cost = kfac.compute_cost(arch, nll_fn, state['w'], X_test, X_test, 
                config['weight_cost'], config['chunk_size'])
            print('Test objective:', cost)
            writer.add_scalar('data/test_objective', cost, i)
            cost = kfac.compute_cost(
                arch, nll_fn, state['w_avg'], X_test, X_test, 
                config['weight_cost'], config['chunk_size'])
            print('Test objective (averaged):', cost)
            writer.add_scalar('data/test_objective_avg', cost, i)

            print()
            cost = kfac.compute_cost(arch, squared_error, state['w'], X_train, X_train, 
                0., config['chunk_size'])
            print('Training error:', cost)
            writer.add_scalar('data/training_error', cost, i)
            cost = kfac.compute_cost(arch, squared_error, state['w_avg'], X_train, X_train, 
                0., config['chunk_size'])
            print('Training error (averaged):', cost)
            writer.add_scalar('data/training_error_avg', cost, i)

            cost = kfac.compute_cost(arch, squared_error, state['w'], X_test, X_test, 
                0., config['chunk_size'])
            print('Test error:', cost)
            writer.add_scalar('data/test_error', cost, i)
            cost = kfac.compute_cost(arch, squared_error, state['w_avg'], X_test, X_test, 
                0., config['chunk_size'])
            print('Test error (averaged):', cost)
            writer.add_scalar('data/test_error_avg', cost, i)
            print()
            

        if i % config['lambda_update_interval'] == 0:
            print('New lambda:', state['lambda'])
            writer.add_scalar('data/lambda', state['lambda'], i)
        if i % config['gamma_update_interval'] == 0:
            print('New gamma:', state['gamma'])
            writer.add_scalar('data/gamma', state['gamma'], i)
        print()

        #print('natgrad_w_pre_norm', state['natgrad_w_pre_norm'])
        #writer.add_scalar('data/natgrad_w_pre_norm', state['natgrad_w_pre_norm'], i)

        #print('natgrad_w_corr_norm', state['natgrad_w_corr_norm'])
        #writer.add_scalar('data/natgrad_w_corr_norm', state['natgrad_w_corr_norm'], i)

        if i % config['cov_update_interval'] == 0:
            plot_matrix_to_tensorboard(writer, config['optimizer'], onp.asarray(state['F_coarse']), 'a. F_coarse', i)
            plot_matrix_to_tensorboard(writer, config['optimizer'], onp.asarray(state['F_hat_coarse']), 'b. F_hat_coarse', i)
            plot_matrix_to_tensorboard(writer, config['optimizer'], onp.abs(state['F_coarse'] - state['F_hat_coarse']), 'c. abs(F_coarse - F_hat_coarse)', i)

        #if (i+1) % 20 == 0:
        #    plot_natgrad_to_tensorboard(writer, onp.asarray(state['natgrad_w_pre']), onp.asarray(state['natgrad_w_corr']), 'kfac natgrad and correction', i)

        if (i+1) % config['conjgrad_benchmark_interval'] == 0:
            plot_conjgrad_convergence_to_tensorboard(writer, state['conjgrad_val'], state['conjgrad_relres'], 'conjgrad convergence plots', i)
            df = get_conjgrad_convergence_dataframe(state['conjgrad_val'], state['conjgrad_relres'])
            dirname = 'cg_benchmark_' + config['experiment']
            os.makedirs(dirname, exist_ok=True)
            df.to_csv(f'{dirname}/iter-{i}.csv')

    writer.close()
