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

import matplotlib.pyplot as plt


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


def train_mlp_drybean():
    """
    """
    lr = 2e-3
    regularization = 1e-4
    hidden_dim = 32

    n_epochs = 50
    batch_size = 32
    
    # dataset = DryBeanDataset(transforms=nn.functional.normalize)
    dataset = DryBeanDataset()
    model = TwoLayerMLP(dataset.get_num_features(), dataset.get_num_classes(), hidden_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=regularization)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1)

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


def main():
    best_model, train_losses, train_accuracies, eval_losses, eval_accuracies = train_mlp_drybean()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(best_model, os.path.join('checkpoints', 'drybean_best_model.pt'))
    plot_training_curves(train_losses[1:], train_accuracies[1:], eval_losses[1:], eval_accuracies[1:],
                         plot_name=os.path.join('checkpoints', 'drybean_loss_curves.png'))

if __name__ == '__main__':
    main()
