import torch.utils
from datasets.uciml import AdultDataset, DryBeanDataset
from models.mlp import TwoLayerMLP, LinearModel
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

import matplotlib.pyplot as plt
import random

def plot_learning_curves(train_losses, eval_losses, plot_name = 'loss_curves.png'):
    """
    Plot the training loss and accuracy curves
    """
    plt.subplot(1, 2, 1)
    for lr, losses in train_losses.items():
        losses = np.clip(losses, None, 100)
        plt.plot(range(len(losses)), losses, label=str(lr))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training')
    plt.subplot(1, 2, 2)
    for lr, losses in eval_losses.items():
        losses = np.clip(losses, None, 100)
        plt.plot(range(len(losses)), losses, label=str(lr))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Testing')
    plt.savefig(plot_name)
    plt.close()


def plot_training_curves(train_losses, train_accuracies, eval_losses, eval_accuracies, plot_name = 'loss_curves.png'):
    """
    Plot the training loss and accuracy curves
    """
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='train_loss')
    plt.plot(range(len(eval_losses)), eval_losses, label='eval_loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accuracies)), train_accuracies, label='train_acc')
    plt.plot(range(len(eval_accuracies)), eval_accuracies, label='eval_acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(plot_name)


def calculate_accuracy(pred : torch.Tensor, target : torch.LongTensor) -> float:
    """
    Calculate accuracy of prediction
    """
    return (torch.sum(torch.argmax(pred, dim=1) == target) / target.shape[0]).item()


def train_model(model : nn.Module, dataloader : DataLoader, criterion : nn.Module, optimizer : Optimizer, device : torch.device) -> tuple:
    """
    """
    temp_losses = []
    temp_accs = []

    model.train()
    for data in dataloader:
        source = data[0].to(device)
        target = data[1].to(device)

        optimizer.zero_grad()
        model.zero_grad()

        output = model(source)
        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        temp_losses.append(loss.item())
        temp_accs.append(calculate_accuracy(output, target))
    
    return (temp_losses, temp_accs)


def eval_model(model : nn.Module, dataloader : DataLoader, criterion : nn.Module, device : torch.device) -> tuple:
    """
    """
    temp_losses = []
    temp_accs = []

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            source = data[0].to(device)
            target = data[1].to(device)

            output = model(source)
            loss = criterion(output, target)

            temp_losses.append(loss.item())
            temp_accs.append(calculate_accuracy(output, target))

    return (temp_losses, temp_accs)


def train_linear_drybean(lr=2e-3, regularization=1e-4, n_epochs=50, batch_size=16):
    """
    """
    dataset = DryBeanDataset()
    model = LinearModel(dataset.get_num_features(), dataset.get_num_classes())

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=regularization)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_set = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    best_acc = 0.
    best_model = None

    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            temp_losses, temp_accs = train_model(model, train_set, criterion, optimizer, device)
            train_losses.append(np.mean(temp_losses))
            train_accuracies.append(np.mean(temp_accs))

            temp_losses, temp_accs = eval_model(model, val_set, criterion, device)
            eval_losses.append(np.mean(temp_losses))
            eval_accuracies.append(np.mean(temp_accs))

            lr_scheduler.step(train_losses[-1])
            
            if eval_accuracies[-1] > best_acc:
                best_acc = eval_accuracies[-1]
                best_model = copy.deepcopy(model)

            pbar.set_postfix(lr = optimizer.param_groups[0]['lr'], train_loss=train_losses[-1],
                             eval_loss=eval_losses[-1], train_acc=train_accuracies[-1], eval_acc=eval_accuracies[-1])
            pbar.update()

    temp_losses, temp_accs = eval_model(best_model, test_set, criterion, device)
    print(f'Final test loss {np.mean(temp_losses)} and accuracy {np.mean(temp_accs)}')
    return best_model, train_losses, train_accuracies, eval_losses, eval_accuracies


def train_mlp_drybean(lr=2e-3, regularization=1e-4, hidden_dim=50, n_epochs=50, batch_size=16):
    """
    """
    dataset = DryBeanDataset()
    model = TwoLayerMLP(dataset.get_num_features(), dataset.get_num_classes(), hidden_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=regularization)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_set = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    best_acc = 0.
    best_model = None

    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            temp_losses, temp_accs = train_model(model, train_set, criterion, optimizer, device)
            train_losses.append(np.mean(temp_losses))
            train_accuracies.append(np.mean(temp_accs))

            temp_losses, temp_accs = eval_model(model, val_set, criterion, device)
            eval_losses.append(np.mean(temp_losses))
            eval_accuracies.append(np.mean(temp_accs))

            lr_scheduler.step(train_losses[-1])
            
            if eval_accuracies[-1] > best_acc:
                best_acc = eval_accuracies[-1]
                best_model = copy.deepcopy(model)

            pbar.set_postfix(lr = optimizer.param_groups[0]['lr'], train_loss=train_losses[-1],
                             eval_loss=eval_losses[-1], train_acc=train_accuracies[-1], eval_acc=eval_accuracies[-1])
            pbar.update()

    temp_losses, temp_accs = eval_model(best_model, test_set, criterion, device)
    print(f'Final test loss {np.mean(temp_losses)} and accuracy {np.mean(temp_accs)}')
    return best_model, train_losses, train_accuracies, eval_losses, eval_accuracies


def train_mlp_adult(lr=1e-3, regularization=1e-4, hidden_dim=25, n_epochs=50, batch_size=16):
    """
    """
    dataset = AdultDataset()
    model = TwoLayerMLP(dataset.get_num_features(), dataset.get_num_classes(), hidden_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=regularization)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.1, 0.2])
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_set = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    best_acc = 0.
    best_model = None

    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            temp_losses, temp_accs = train_model(model, train_set, criterion, optimizer, device)
            train_losses.append(np.mean(temp_losses))
            train_accuracies.append(np.mean(temp_accs))

            temp_losses, temp_accs = eval_model(model, val_set, criterion, device)
            eval_losses.append(np.mean(temp_losses))
            eval_accuracies.append(np.mean(temp_accs))

            lr_scheduler.step(train_losses[-1])
            
            if eval_accuracies[-1] > best_acc:
                best_acc = eval_accuracies[-1]
                best_model = copy.deepcopy(model)

            pbar.set_postfix(lr = optimizer.param_groups[0]['lr'], train_loss=train_losses[-1],
                             eval_loss=eval_losses[-1], train_acc=train_accuracies[-1], eval_acc=eval_accuracies[-1])
            pbar.update()

    temp_losses, temp_accs = eval_model(best_model, test_set, criterion, device)
    print(f'Final test loss {np.mean(temp_losses)} and accuracy {np.mean(temp_accs)}')
    return best_model, train_losses, train_accuracies, eval_losses, eval_accuracies


def set_seed(seed = 123456789):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # This environment variable is required for `use_deterministic_algorithms`
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def main():
    set_seed()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    # Best model
    best_model, train_losses, train_accuracies, eval_losses, eval_accuracies = train_linear_drybean(lr=1e-2, batch_size=8)

    training_accuracy = {}
    eval_accuracy = {}
    learning_rates = [2e-3, 1e-3, 5e-3, 1e-2]
    for lr in learning_rates:
        set_seed()
        _,_,train_accuracies,_,eval_accuracies= train_mlp_drybean(lr=lr)
        training_accuracy[lr] = train_accuracies
        eval_accuracy[lr] = eval_accuracies
    plot_learning_curves(training_accuracy, eval_accuracy, plot_name=os.path.join('checkpoints', 'drybean_nn_lr_loss_curves.png'))

    training_accuracy = {}
    eval_accuracy = {}
    batch_sizes = [8, 16, 32, 64]
    for bs in batch_sizes:
        set_seed()
        _,_,train_accuracies,_,eval_accuracies= train_mlp_drybean(batch_size=bs)
        training_accuracy[bs] = train_accuracies
        eval_accuracy[bs] = eval_accuracies
    plot_learning_curves(training_accuracy, eval_accuracy, plot_name=os.path.join('checkpoints', 'drybean_nn_bs_loss_curves.png'))

    training_accuracy = {}
    eval_accuracy = {}
    learning_rates = [2e-3, 1e-3, 5e-3, 1e-2]
    for lr in learning_rates:
        set_seed()
        _,_,train_accuracies,_,eval_accuracies= train_mlp_adult(lr=lr)
        training_accuracy[lr] = train_accuracies
        eval_accuracy[lr] = eval_accuracies
    plot_learning_curves(training_accuracy, eval_accuracy, plot_name=os.path.join('checkpoints', 'adult_nn_lr_loss_curves.png'))

    training_accuracy = {}
    eval_accuracy = {}
    batch_sizes = [8, 16, 32, 64]
    for bs in batch_sizes:
        set_seed()
        _,_,train_accuracies,_,eval_accuracies= train_mlp_adult(batch_size=bs)
        training_accuracy[bs] = train_accuracies
        eval_accuracy[bs] = eval_accuracies
    plot_learning_curves(training_accuracy, eval_accuracy, plot_name=os.path.join('checkpoints', 'adult_nn_bs_loss_curves.png'))


if __name__ == '__main__':
    main()
