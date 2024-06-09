from datasets.uciml import AdultDataset, DryBeanDataset
import numpy as np
import os
import pickle
import copy

from torch import manual_seed, use_deterministic_algorithms
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from sklearn.experimental import enable_halving_search_cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score, StratifiedShuffleSplit


def plot_training_curves(train_sizes, eval_accuracies, test_accuracies, plot_name = 'loss_curves.png'):
    """
    Plot the training loss and accuracy curves
    """
    plt.subplot(1, 2, 1)
    for idx,size in enumerate(train_sizes):
        plt.plot(eval_accuracies[idx], label=str(size))
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training')
    plt.subplot(1, 2, 2)
    plt.plot(train_sizes, test_accuracies)
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Testing')
    plt.savefig(plot_name)
    plt.close()

def calculate_accuracy(pred : np.ndarray, target : np.ndarray) -> float:
    """
    Calculate accuracy of prediction
    """
    return (np.sum(pred == target) / target.shape[0]).item()


def train_knn_drybean(p_grid={}, test_size=0.2, eval_size=0.3):
    """
    """
    n_epochs = 5

    dataset = DryBeanDataset()
    train_set, test_set = random_split(dataset, [1.0 - test_size, test_size])

    model = KNeighborsClassifier(weights='distance')
    best_model = None

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    # Arrays to store scores
    non_nested_scores = []
    nested_scores = []

    for i in tqdm(range(n_epochs)):
        inner_cv = StratifiedShuffleSplit(n_splits=3, test_size=eval_size, random_state=i)
        outer_cv = StratifiedShuffleSplit(n_splits=3, test_size=eval_size, random_state=i)

        # Non_nested parameter search and scoring
        clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=outer_cv, scoring='accuracy', refit=True, n_jobs=-1)
        clf.fit(X, y)
        non_nested_scores.append(clf.best_score_)

        # Nested CV with parameter optimization
        nested_clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, scoring='accuracy', refit=True, n_jobs=-1)
        nested_score = cross_val_score(nested_clf, X=X, y=y, cv=outer_cv)

        if len(nested_scores) == 0 or nested_score.mean() > nested_scores[-1]:
            print(clf.best_params_)
            best_model = copy.deepcopy(clf.best_estimator_)

        nested_scores.append(nested_score.mean())

    X, y = test_set[:]
    X = X.numpy()
    y = y.numpy()
    pred = best_model.predict(X)

    acc = calculate_accuracy(pred, y)
    return best_model, nested_scores, acc


def train_knn_adult(p_grid={}, test_size=0.2, eval_size=0.3):
    """
    """
    n_epochs = 3

    dataset = AdultDataset()
    train_set, test_set = random_split(dataset, [1.0 - test_size, test_size])

    model = KNeighborsClassifier(weights='distance')
    best_model = None

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    # Arrays to store scores
    non_nested_scores = []
    nested_scores = []

    for i in tqdm(range(n_epochs)):
        inner_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=i)
        outer_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=i)

        # Non_nested parameter search and scoring
        clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=outer_cv, scoring='accuracy', refit=True, n_jobs=3)
        clf.fit(X, y)
        non_nested_scores.append(clf.best_score_)

        # Nested CV with parameter optimization
        nested_clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, scoring='accuracy', refit=True, n_jobs=3)
        nested_score = cross_val_score(nested_clf, X=X, y=y, cv=outer_cv)

        if len(nested_scores) == 0 or nested_score.mean() > nested_scores[-1]:
            print(clf.best_params_)
            best_model = copy.deepcopy(clf.best_estimator_)

        nested_scores.append(nested_score.mean())

    X, y = test_set[:]
    X = X.numpy()
    y = y.numpy()
    pred = best_model.predict(X)

    acc = calculate_accuracy(pred, y)
    return best_model, nested_scores, acc


def set_seed(seed = 123456789):
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    # This environment variable is required for `use_deterministic_algorithms`
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    use_deterministic_algorithms(True)


def main():
    set_seed()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    train_sizes = []
    test_accuracies = []
    eval_accuracies = []
    test_sizes = np.arange(0.1, 1.0, 0.2)
    # Cs = []
    # C_space = np.logspace(-2, 3, 5)
    # test_sizes = [0.3]
    for size in test_sizes:
        # for c in C_space:
        set_seed()
        _, eval_accuracy, test_accuracy = train_knn_drybean(p_grid={'n_neighbors' : np.arange(5, 10), 'leaf_size' : np.logspace(8, 12, 10, base=2, dtype=int)})
        # Cs.append(c)
        train_sizes.append(round(1.0 - size, 1))
        test_accuracies.append(test_accuracy)
        eval_accuracies.append(eval_accuracy)
    plot_training_curves(train_sizes, eval_accuracies, test_accuracies, os.path.join('checkpoints', 'drybean_knn_learning_curves.png'))

    # best_model, eval_accuracies, test_accuracy = train_knn_drybean()
    # print(f'Final test accuracy for kNN {test_accuracy}')
    # with open(os.path.join('checkpoints', 'drybean_knn_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'drybean_knn_acc_curves.png'))
    
    # best_model, eval_accuracies, test_accuracy = train_knn_adult(p_grid = {'n_neighbors' : np.arange(5, 10), 'leaf_size' : np.logspace(8, 12, 10, base=2, dtype=int)})
    # print(f'Final test accuracy for kNN {test_accuracy}')
    # with open(os.path.join('checkpoints', 'adult_knn_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'adult_knn_acc_curves.png'))


if __name__ == '__main__':
    main()
