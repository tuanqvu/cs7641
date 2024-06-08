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
from sklearn.manifold import TSNE
from sklearn.svm import SVC, LinearSVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, HalvingGridSearchCV, cross_val_score, StratifiedShuffleSplit

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


def train_svm_drybean(kernel='linear', p_grid={}, n_splits=3, test_size=0.3, eval_size=0.3, n_jobs=-1, verbose=0):
    """
    """
    n_epochs = 2

    dataset = DryBeanDataset()
    train_set, test_set = random_split(dataset, [1.0 - test_size, test_size])

    if kernel == 'linear':
      model = LinearSVC(dual='auto')
    else:
      model = SVC(kernel=kernel)
    best_model = None

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    # Arrays to store scores
    non_nested_scores = []
    nested_scores = []

    for i in tqdm(range(n_epochs)):
        inner_cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=eval_size, random_state=i)
        outer_cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=eval_size, random_state=i)

        # Non_nested parameter search and scoring
        clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=outer_cv, min_resources='smallest',
                                  scoring='accuracy', n_jobs=n_jobs, verbose=verbose)
        clf.fit(X, y)
        non_nested_scores.append(clf.best_score_)

        # Nested CV with parameter optimization
        nested_clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, min_resources='smallest',
                                         scoring='accuracy', n_jobs=n_jobs, verbose=verbose)
        nested_score = cross_val_score(nested_clf, X=X, y=y, cv=outer_cv, n_jobs=n_jobs, verbose=verbose)

        if len(nested_scores) == 0 or nested_score.mean() > nested_scores[-1]:
            print(clf.best_params_, clf.best_score_)
            best_model = copy.deepcopy(clf.best_estimator_)

        nested_scores.append(nested_score.mean())

    X, y = test_set[:]
    X = X.numpy()
    y = y.numpy()
    pred = best_model.predict(X)

    acc = calculate_accuracy(pred, y)
    return best_model, nested_scores, acc


def train_linear_svm_drybean(n_splits=3, test_size=0.2, eval_size=0.3):
    """
    """
    dataset = DryBeanDataset()
    train_set, test_set = random_split(dataset, [1.0 - test_size, test_size])

    model = SVC(kernel='linear', C=0.8, gamma='scale', class_weight='balanced')
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=eval_size)

    best_model = None
    eval_accuracies = []

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    with tqdm(total=skf.get_n_splits(X, y)) as pbar:
        temp_val_accs = []
        for _, (train_index, test_index) in enumerate(skf.split(X, y)):
            x_train, y_train = X[train_index], y[train_index]
            model.fit(x_train, y_train)
 
            x_val, y_val = X[test_index], y[test_index]
            pred = model.predict(x_val)
            acc = calculate_accuracy(pred, y_val)
            temp_val_accs.append(acc)

            if len(eval_accuracies) == 0 or np.mean(temp_val_accs) > eval_accuracies[-1]:
                best_model = copy.deepcopy(model)

            eval_accuracies.append(np.mean(temp_val_accs))
            pbar.set_postfix(val_acc=eval_accuracies[-1])
            pbar.update()

    X, y = test_set[:]
    X = X.numpy()
    y = y.numpy()
    pred = best_model.predict(X)

    acc = calculate_accuracy(pred, y)
    return best_model, eval_accuracies, acc


def train_svm_adult(kernel='linear', p_grid={}, n_splits=3, test_size=0.3, eval_size=0.2, n_jobs=-1, verbose=0):
    """
    """
    n_epochs = 2

    dataset = AdultDataset()
    train_set, test_set = random_split(dataset, [1.0 - test_size, test_size])

    if kernel == 'linear':
      model = LinearSVC(dual='auto')
    else:
      model = SVC(kernel=kernel)
    best_model = None

    X, y = train_set[:]
    X = X.numpy()
    y = y.numpy()

    # Arrays to store scores
    non_nested_scores = []
    nested_scores = []

    for i in tqdm(range(n_epochs)):
        inner_cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=eval_size, random_state=i)
        outer_cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=eval_size, random_state=i)

        # Non_nested parameter search and scoring
        clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=outer_cv, min_resources='smallest',
                                  scoring='accuracy', n_jobs=n_jobs, verbose=verbose)
        clf.fit(X, y)
        non_nested_scores.append(clf.best_score_)

        # Nested CV with parameter optimization
        nested_clf = HalvingGridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv, min_resources='smallest',
                                         scoring='accuracy', n_jobs=n_jobs, verbose=verbose)
        nested_score = cross_val_score(nested_clf, X=X, y=y, cv=outer_cv, n_jobs=n_jobs, verbose=verbose)

        if len(nested_scores) == 0 or nested_score.mean() > nested_scores[-1]:
            print(clf.best_params_, clf.best_score_)
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
    
    dataset = DryBeanDataset()
    dataset_length = len(dataset)

    train_sizes = []
    test_accuracies = []
    eval_accuracies = []
    test_sizes = np.arange(0.1, 1.0, 0.2)
    for size in test_sizes:
      print(size)
      _, eval_accuracy, test_accuracy = train_svm_drybean(kernel='linear', test_size=size, p_grid={'class_weight' : [None], 
                                                                                                   'dual' : ['auto'],
                                                                                                   'multi_class' : ['ovr'],
                                                                                                   'max_iter' : [1000000],
                                                                                                   'C' : np.logspace(-2, 3, 10)}, verbose=0)
      train_sizes.append(int((1.0 - size) * dataset_length))
      test_accuracies.append(test_accuracy)
      eval_accuracies.append(eval_accuracy)
    plot_training_curves(train_sizes, eval_accuracies, test_accuracies, os.path.join('checkpoints', 'drybean_svm_linear_acc_curves.png'))

    # best_model, eval_accuracies, test_accuracy = train_linear_svm_drybean()
    # best_model, eval_accuracies, test_accuracy = train_svm_drybean(kernel='linear', p_grid={'class_weight' : [None, 'balanced'], 
    #                                                                                         'dual' : ['auto'],
    #                                                                                         'multi_class' : ['ovr', 'crammer_singer'],
    #                                                                                         'max_iter' : [1000000],
    #                                                                                         'C' : np.logspace(-2, 3, 10)}, verbose=2)
    # print(f'Final test accuracy for SVM with linear kernel {test_accuracy}')
    # with open(os.path.join('checkpoints', 'drybean_svm_linear_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'drybean_svm_linear_acc_curves.png'))

    # best_model, eval_accuracies, test_accuracy = train_svm_drybean(kernel='rbf', p_grid={'C' : np.logspace(3, 4, 10), 'gamma' : np.logspace(-6, -4, 10)})
    # print(f'Final test accuracy for SVM with rbf kernel {test_accuracy}')
    # with open(os.path.join('checkpoints', 'drybean_svm_rbf_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'drybean_svm_rbf_acc_curves.png'))

    # best_model, eval_accuracies, test_accuracy = train_svm_adult(kernel='linear', p_grid={'class_weight' : [None, 'balanced'], 
    #                                                                                       'dual' : ['auto'],
    #                                                                                       'multi_class' : ['ovr', 'crammer_singer'],
    #                                                                                       'max_iter' : [1000000],
    #                                                                                       'C' : np.logspace(-2, 3, 10)}, verbose=2)
    # print(f'Final test accuracy for SVM with linear kernel {test_accuracy}')
    # with open(os.path.join('checkpoints', 'adult_svm_linear_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'adult_svm_linear_acc_curves.png'))
    
    # best_model, eval_accuracies, test_accuracy = train_svm_adult(kernel='rbf', p_grid={'C' : np.logspace(3, 4, 10), 'gamma' : np.logspace(-6, -4, 10)})
    # print(f'Final test accuracy for SVM with rbf kernel {test_accuracy}')
    # with open(os.path.join('checkpoints', 'adult_svm_rbf_model.pkl'), 'wb') as f:
    #     pickle.dump(best_model, f, protocol=5)
    # plot_training_curves(eval_accuracies, plot_name=os.path.join('checkpoints', 'adult_svm_rbf_acc_curves.png'))


def plot_tsne(X, y, title, plot_name):
    tsne = TSNE(n_components=2, random_state=0, n_jobs=-1, verbose=2)
    emb = tsne.fit_transform(X, y)
    plt.figure()
    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False)
    plt.title(title)
    plt.scatter(emb[:,0], emb[:,1], c=y)
    plt.savefig(f"{plot_name}.jpg")
    plt.close()


if __name__ == '__main__':
    main()
