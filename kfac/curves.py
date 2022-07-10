import argparse
import scipy.io

import autoencoders


def get_architecture():
    layer_sizes = [('enc1', 400),
                   ('enc2', 200),
                   ('enc3', 100),
                   ('enc4', 50),
                   ('enc5', 25),
                   ('code', 6),
                   ('dec1', 25),
                   ('dec2', 50),
                   ('dec3', 100),
                   ('dec4', 200),
                   ('dec5', 400)]

    return autoencoders.get_architecture(784, layer_sizes)

def get_config():
    return autoencoders.default_config()

def run(args):
    try:
        obj = scipy.io.loadmat('digs3pts_1.mat')
    except:
        print("To run this script, first download https://www.cs.toronto.edu/~jmartens/digs3pts_1.mat to this directory.")

    X_train = obj['bdata']
    X_test = obj['bdatatest']

    config = get_config()
    arch = get_architecture()

    config['experiment'] = 'curves'
    config['optimizer'] = args.optimizer
    config['comment'] = args.comment
    config['random_seed'] = args.random_seed
    config['use_momentum'] = args.use_momentum
    config['init_lambda'] = args.init_lambda
    config['adapt_gamma'] = args.adapt_gamma
    config['conjgrad_benchmark_interval'] = args.conjgrad_benchmark_interval
    config['nbasis'] = args.nbasis
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
    args = parser.parse_args()
    run(args)


