import os

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import models
from torchvision.utils import make_grid
from torchvision import transforms as tsfm
from torchvision.datasets import ImageFolder

import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from IPython import display

from model import VGG16
from predict import predict, view_pred_result
from train import train
from valid import valid
from dataset import Train_data, Pred_data


def Plot(title, ylabel, epochs, train_loss, valid_loss):
    plt.figure()
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel(ylabel)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.legend(['train', 'valid'], loc='upper left')


if __name__ == '__main__':
    data_dir = 'plant_seedlings_classification/'  # dataset's dir you want to unzip
    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.backends.cudnn.deterministic = True

    # Set Hyperparameters
    batch_size = 64
    epochs = 1  # 50
    learning_rate = 0.001
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initial transform
    transform = tsfm.Compose([
        tsfm.Resize((224, 224)),
        tsfm.ToTensor(),
    ])

    # initial dataset
    whole_set = Train_data(
        root_dir=train_dir,
        transform=transform
    )

    test_set = Pred_data(
        root_dir=test_dir,
        transform=transform
    )

    # split train valid and initial dataloader
    train_set_size = int(len(whole_set) * 0.8)
    valid_set_size = len(whole_set) - train_set_size
    train_set, valid_set = random_split(whole_set, [train_set_size, valid_set_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    # initial model
    model = VGG16(num_classes=12).to(device)

    # initial loss_function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # initial plot values
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    epoch_list = []

    # repeat train and valid epochs times
    print(epochs)
    for epoch in range(epochs):
        epoch_list.append(epoch + 1)

        loss, acc = train(
            device,
            model,
            criterion,
            optimizer,
            train_loader,
            epoch=epoch,
            total_epochs=epochs,
            batch_size=batch_size
        )
        train_loss.append(loss)
        train_acc.append(acc)
        print(f'Avg train Loss: {loss}, Avg train acc: {acc}')

        loss, acc = valid(
            device,
            model,
            criterion,
            valid_loader,
            epoch=epoch,
            total_epochs=epochs,
            batch_size=batch_size
        )
        valid_loss.append(loss)
        valid_acc.append(acc)
        print(f'Avg valid Loss: {loss}, Avg valid acc: {acc}')

    Plot("Loss Curve", 'Loss', epoch_list, train_loss, valid_loss)
    Plot("Accuracy Curve", 'Acc', epoch_list, train_acc, valid_acc)

    preds = predict(test_set, model)
    view_pred_result(test_set, preds)
