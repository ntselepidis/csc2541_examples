import argparse
import scipy.io

import autoencoders


def get_architecture():
    layer_sizes = [('enc1', 400),
                   ('enc2', 350),
                   ('enc3', 300),
                   ('enc4', 250),
                   ('enc5', 200),
                   ('enc6', 150),
                   ('enc7', 100),
                   ('enc8', 50),
                   ('enc9', 25),
                   ('code', 6),
                   ('dec1', 25),
                   ('dec2', 50),
                   ('dec3', 100),
                   ('dec4', 150),
                   ('dec5', 200),
                   ('dec6', 250),
                   ('dec7', 300),
                   ('dec8', 350),
                   ('dec9', 400)]

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

    config['experiment'] = 'curves-deep'
    config['optimizer'] = args.optimizer
    config['comment'] = args.comment
    config['random_seed'] = args.random_seed
    autoencoders.run_training(X_train, X_test, arch, config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='kfac', type=str)
    parser.add_argument('--comment', default='default', type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    args = parser.parse_args()
    run(args)

