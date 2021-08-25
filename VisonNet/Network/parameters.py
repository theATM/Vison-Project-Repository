

#Training Parameters
MAX_EPOCH_NUMBER = 105
TRAIN_ARCH = 'cuda'  # for cpu type 'cpu', for gpu type 'cuda'


#Data Parameters
BEST_MODEL_PATH = '../Models/best_model.pth'
DATASET_PATH = '../Data/dataset'
TESTSET_PATH = '../Data/testset'
VALSET_PATH = '../Data/valset'

#Quantisation Parameters


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def xyz():
    model = models.resnet50()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    dataset = datasets.FakeData(
        size=1000,
        transform=transforms.ToTensor())
    loader = DataLoader(
        dataset,
        num_workers=0,
        pin_memory=True
    )

    model.to('cuda')


    for data, target in loader:
        data = data.to('cuda', non_blocking=True)
        target = target.to('cuda', non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
