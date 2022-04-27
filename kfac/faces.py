import argparse
import scipy.io

import autoencoders


def get_architecture():
    layer_sizes = [('enc1', 2000),
                   ('enc2', 1000),
                   ('enc3', 500),
                   ('code', 30),
                   ('dec1', 500),
                   ('dec2', 1000),
                   ('dec3', 2000)]

    return autoencoders.get_architecture(625, layer_sizes)

def get_config():
    return autoencoders.default_config()

def run(args):
    try:
        obj = scipy.io.loadmat('newfaces_rot_single.mat')
    except:
        print("To run this script, first download https://www.cs.toronto.edu/~jmartens/newfaces_rot_single.mat to this directory.")

    total = 165600
    trainsize = (total//40)*25
    testsize = (total//40)*10

    X = obj['newfaces_single'].T
    X_train = X[0:trainsize, :]
    X_test = X[(total-testsize):total, :]

    config = get_config()
    arch = get_architecture()

    config['experiment'] = 'faces'
    config['optimizer'] = args.optimizer
    config['comment'] = args.comment
    config['random_seed'] = args.random_seed
    config['use_momentum'] = args.use_momentum
    autoencoders.run_training(X_train, X_test, arch, config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='kfac', type=str)
    parser.add_argument('--comment', default='default', type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--use_momentum', default=1, type=int, choices=[0, 1])
    args = parser.parse_args()
    run(args)


