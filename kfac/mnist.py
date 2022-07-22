import argparse
import numpy as np
import tensorflow_datasets as tfds

import autoencoders



## def MNISTArchitecture():
##     net_init, net_apply = named_serial(
##         ('enc1z', Dense(1000, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('enc1a', elementwise(nn.sigmoid)),
##         ('enc2z', Dense(500, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('enc2a', elementwise(nn.sigmoid)),
##         ('enc3z', Dense(250, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('enc3a', elementwise(nn.sigmoid)),
##         ('code', Dense(30, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('dec1z', Dense(250, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('dec1a', elementwise(nn.sigmoid)),
##         ('dec2z', Dense(500, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('dec2a', elementwise(nn.sigmoid)),
##         ('dec3z', Dense(1000, W_init=sparse_init(), b_init=initializers.zeros)),
##         ('dec3a', elementwise(nn.sigmoid)),
##         ('out', Dense(784, W_init=sparse_init(), b_init=initializers.zeros)),
##     )
##     param_info = (('in', 'enc1z'),
##                   ('enc1a', 'enc2z'),
##                   ('enc2a', 'enc3z'),
##                   ('enc3a', 'code'),
##                   ('code', 'dec1z'),
##                   ('dec1a', 'dec2z'),
##                   ('dec2a', 'dec3z'),
##                   ('dec3a', 'out')
##                  )
##     in_shape=(-1, 784)
##     flatten, unflatten = get_flatten_fns(net_init, in_shape)
##     return Architecture(net_init, net_apply, in_shape, flatten, unflatten, param_info)

def get_architecture():
    layer_sizes = [('enc1', 1000),
                   ('enc2', 500),
                   ('enc3', 250),
                   ('code', 30),
                   ('dec1', 250),
                   ('dec2', 500),
                   ('dec3', 1000)]

    return autoencoders.get_architecture(784, layer_sizes)


def get_config():
    return autoencoders.default_config()




def run(args):
    mnist_data, info = tfds.load(name="mnist", batch_size=-1, with_info=True)
    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data['train'], mnist_data['test']
    X_train = train_data['image'].reshape((-1, 784)).astype(np.float32) / 255
    X_test = test_data['image'].reshape((-1, 784)).astype(np.float32) / 255

    config = get_config()
    arch = get_architecture()

    config['experiment'] = 'mnist'
    config['optimizer'] = args.optimizer
    config['comment'] = args.comment
    config['random_seed'] = args.random_seed
    config['use_momentum'] = args.use_momentum
    config['init_lambda'] = args.init_lambda
    config['adapt_gamma'] = args.adapt_gamma
    config['conjgrad_benchmark_interval'] = args.conjgrad_benchmark_interval
    config['nbasis'] = args.nbasis
    config['conjgrad_maxiter'] = args.conjgrad_maxiter
    autoencoders.run_training(X_train, X_test, arch, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='kfac', type=str)
    parser.add_argument('--comment', default='default', type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--use_momentum', default=1, type=int, choices=[0, 1])
    parser.add_argument('--init_lambda', default=150, type=float)
    parser.add_argument('--adapt_gamma', default=1, type=int, choices=[0, 1])
    parser.add_argument('--conjgrad_benchmark_interval', default=20010, type=int)
    parser.add_argument('--nbasis', default=1, type=int)
    parser.add_argument('--conjgrad_maxiter', default=5, type=int)
    args = parser.parse_args()
    run(args)


