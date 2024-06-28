import torch.utils
from datasets.uciml import AdultDataset, DryBeanDataset
from models.mlp import TwoLayerMLP
from tqdm import tqdm
import numpy as np
import copy
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

import matplotlib as mpl
import matplotlib.pyplot as plt
import random

import itertools
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection


def set_seed(seed=123456789):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # This environment variable is required for `use_deterministic_algorithms`
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def plot_results(scores, plot_name=''):
    xticks = None
    for name, item in scores.items():
        if name == 'Ground truth':
            continue
        else:
            x = np.asarray(item)[:, 0].astype(int)
            y = np.asarray(item)[:, 1]
            if xticks is None:
                xticks = x
            plt.plot(x, y, label=name)
    plt.axhline(scores['Ground truth'], linestyle='--', color='red', label='Ground truth')
    plt.xticks(xticks)
    plt.grid(visible=True)
    plt.legend()
    plt.title('Clustering')
    plt.ylabel('Silhouette Coeff.')
    plt.xlabel('Number of Clusters')
    plt.savefig(plot_name)
    plt.close()


def clustering():
    datasets = [DryBeanDataset(), AdultDataset()]
    dataset_names = ['drybean', 'adult']

    for dataset, dataset_name in zip(datasets, dataset_names):
        scores = {}

        set_seed()
        use_set, _ = random_split(dataset, [0.3, 0.7])
        X, y = use_set[:]
        X = X.numpy()
        y = y.numpy()

        score = silhouette_score(X, y, random_state=0)
        scores['Ground truth'] = [score]

        n_clusters = np.arange(2, 12, dtype=int)
        for n in tqdm(n_clusters):
            name = 'Mixture of Gaussian'
            gm = GaussianMixture(
                n_components=n, random_state=0, verbose=0).fit(X)
            score = silhouette_score(X, gm.predict(X), random_state=0)
            if name not in scores:
                scores[name] = []
            scores[name].append([n, score])

            name = 'KMeans'
            km = KMeans(n_clusters=n, random_state=0).fit(X)
            score = silhouette_score(X, km.predict(X), random_state=0)
            if name not in scores:
                scores[name] = []
            scores[name].append([n, score])

        plot_results(scores, os.path.join(
            'checkpoints', f"{dataset_name}_clustering.png"))


def dimension_reduction():
    datasets = [DryBeanDataset(), AdultDataset()]
    dataset_names = ['drybean', 'adult']

    for dataset, dataset_name in zip(datasets, dataset_names):
        scores = {}

        set_seed()
        use_set, _ = random_split(dataset, [0.3, 0.7])
        X, y = use_set[:]
        X = X.numpy()
        y = y.numpy()


def main():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    print("Running clustering")
    clustering()


if __name__ == '__main__':
    main()
