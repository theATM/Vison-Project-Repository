import os
import torch
import torchvision
import time
import copy
import torch.optim as optim

import random

from torch import nn
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import MobileNetV2
from torch.optim import lr_scheduler
from PIL import Image

import bankset as bank
import parameters as par
import model as mod


def main():

    # Load Data
    trainloader, valloader, testloader = loadData()

    # Creating devices  - choosing where will training/eval calculate (gpu or cpu)
    trainDevice = torch.device(par.TRAIN_ARCH)
    evalDevice = torch.device(par.TRAIN_ARCH)

    if par.TRAIN_ARCH == 'cuda:0': torch.cuda.empty_cache()


    # Prepare Model
    model = mod.create(load=par.LOAD_MODEL, loadPath=par.BEST_MODEL_PATH)
    model.to(trainDevice)

    # Create Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=par.INITIAl_LEARNING_RATE)

    # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[4, 15, 20, 30, 40, 95], gamma=0.8)

    best_acc = 0

    # Training Network
    print('Training Started')
    #training_start_time = time.time()
    for nEpoch in range(par.MAX_EPOCH_NUMBER):
        print('\n' + 'Epoch ' + str(nEpoch) + ':')
        # Training Model
        model.train()
        train_one_epoch(model, criterion, optimizer, trainloader, trainDevice)

        # Stepping scheduler
        exp_lr_scheduler.step()

        # Evaluate in some epochs:
        if nEpoch % par.EVAL_PER_EPOCHS == 0:
            model.eval()
            mtop1, mtop5 = evaluate(model, criterion, valloader, evalDevice)
            # Save If Best
            if mtop1.avg > best_acc:
                best_acc = mtop1.avg
                saveModel(model, best_acc)
            if evalDevice == 'cuda:0' : torch.cuda.empty_cache()
            print('Evaluation on epoch %d accuracy on all validation images, %2.2f' % (nEpoch, mtop1.avg))

    print('Epoch ' + str(nEpoch) + ' completed')

    #Post Training Evaluation
    model.eval()
    xtop1, xtop5 = evaluate(model, criterion, testloader, evalDevice)
    print('\n' + 'Evaluation accuracy on all test images, %2.2f' % (xtop1.avg))
    print("Finished Training")

    #Save If Best #TODO is this really comparable to valset?
    if xtop1.avg > best_acc:
        best_acc = mtop1.avg
        saveModel(model, best_acc)


def loadData(singleBatchTest = False):

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        bank.RandomRotationTransform(angles=[-90, 90, 0, 180, -180]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.75, 1.45), contrast=0.5, saturation=0.5, hue=0.3),
        torchvision.transforms.RandomApply(
            [torchvision.transforms.Grayscale(num_output_channels=3)],
            p=0.35
        ),
        transforms.ToTensor(),
        torchvision.transforms.RandomApply(
            [bank.AddGaussianNoise(0., 1.)],
            p=0.45
        ),
        transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.08), ratio=(0.5, 2.3), value='random'),
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703])
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703])
    ])

    # Load Data
    trainset = bank.Bankset(par.DATASET_PATH, transform_train)
    trainloader = DataLoader(trainset, batch_size=6, shuffle=True, pin_memory=True, num_workers=4)

    valset = bank.Bankset(par.VALSET_PATH, transform_val)
    valloader = DataLoader(valset, batch_size=4, shuffle=True, num_workers=2)

    testset = bank.Bankset(par.TESTSET_PATH, transform_test)
    testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
    print("Data Loaded")

    if singleBatchTest == True:
        #Preform Single Batch Test
        singleBatch = next(iter(trainloader))
        singleBatchImages, singleBatchLabels, singleBatchNames = singleBatch['image'], singleBatch['class'], singleBatch['name']
        singleBatchDataSet = bank.SingleBatchTestSet(singleBatchImages, singleBatchLabels, singleBatchNames)
        trainloader = DataLoader(singleBatchDataSet, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)

    return trainloader, valloader, testloader


def train_one_epoch(model, criterion, optimizer, data_loader, trainDevice):

    # Defines statistical variables
    top1 = bank.AverageMeter('Accuracy', ':6.2f')
    top3 = bank.AverageMeter('In Top 3', ':6.2f')
    avgLoss = bank.AverageMeter('Loss', '1.5f')
    epochStartTime = time.time()

    # Training Loop (Through all data pictures)
    for i, data in enumerate(data_loader):
        inputs = torch.autograd.Variable(data['image'].to(trainDevice, non_blocking=True))
        labels = torch.autograd.Variable(data['class'].to(trainDevice, non_blocking=True))
        # Calculate Network Function (what Network thinks of this image)
        output = model(inputs)
        # Calculate loss
        loss = criterion(output, labels)
        # Resets Gradient to Zeros (clearing it before using in calculations)
        optimizer.zero_grad()
        # Backpropagate loss
        loss.backward()
        #Clipping the gradient #TODO test
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
        # Update the weighs
        optimizer.step()
        # Calculate Accuracy
        acc1, acc3 = accuracy(output, labels, topk=(1, 3))
        # Update Statistics
        top1.update(acc1[0], inputs.size(0))
        top3.update(acc3[0], inputs.size(0))
        avgLoss.update(loss, inputs.size(0))
        if i % 1000 == 0:
            print('Image ' + str(i) + ' Current Loss {loss:.3f}'
                  .format(loss=loss.item()))

    # Print Result for One Epoch of Training
    print('Full train set:  * Accuracy {top1.avg:.3f} | In Top 3 {top3.avg:.3f} | Loss {avgLoss.avg:.3f} | Used Time {epochTime:.2f} s'
          .format(top1=top1, top3=top3, avgLoss=avgLoss,epochTime = time.time() - epochStartTime))


def saveModel(model, best_acc):
    best_model = copy.deepcopy(model.state_dict())
    torch.save(best_model, par.BEST_MODEL_PATH)
    print("Saved ", best_acc)


def evaluate(model, criterion, data_loader, device):

    top1 = bank.AverageMeter('Accuracy', ':6.2f')
    top3 = bank.AverageMeter('In Top 3', ':6.2f')

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = data['image'].to(device)
            labels = data['class'].to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top3.update(acc5[0], inputs.size(0))

    return top1, top3


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad(): # disables recalculation of gradients
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    #Run main function
    main()










