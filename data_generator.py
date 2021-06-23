import argparse

import pandas as pd

from sklearn.datasets import make_moons, make_circles, make_blobs


def get_dataset(random_state, n_samples, generate_type='moons'):
    data, target = None, None
    if generate_type == 'moons':
        data, target = make_moons(noise=0.09, random_state=random_state, n_samples=n_samples)
    elif generate_type == 'blobs':
        data, target = make_blobs(random_state=random_state, n_samples=n_samples, centers=2)
    elif generate_type == 'circles':
        data, target = make_circles(noise=0.09, random_state=random_state, n_samples=n_samples, factor=0.5)
    return data, target


def data_generator(name, data_type, n_samples, random_seed):
    data, target = get_dataset(random_seed, n_samples, data_type)
    if target is None:
        raise RuntimeError('Unknown data type')

    res = pd.DataFrame()
    res['target'] = target
    res['f1'] = data[:, 0]
    res['f2'] = data[:, 1]
    res.to_csv(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset')
    parser.add_argument('name', type=str)
    parser.add_argument('data_type', type=str)
    parser.add_argument('n_samples', type=int)
    parser.add_argument('random_seed', type=int)

    args = parser.parse_args()
    data_generator(args.name, args.data_type, args.n_samples, args.random_seed)
