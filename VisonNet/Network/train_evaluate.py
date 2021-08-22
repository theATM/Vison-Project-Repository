import os

import torch
import torchvision
import time
import copy
import torch.optim as optim
import torchvision.models.quantization as models
import random

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import MobileNetV2

from Bankset import Bankset
from torch.optim import lr_scheduler
from PIL import Image

MAX_EPOCH_NUMBER = 105
TRAIN_ARCH = 'cuda'  # for cpu type 'cpu', for gpu type 'cuda'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def create_combined_model(model_fe):
    model_fe_features = nn.Sequential(
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        model_fe.avgpool,
    )

    new_head = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(256, 7),
    )

    new_model = nn.Sequential(
        model_fe_features,
        nn.Flatten(1),
        new_head,
    )
    return new_model


def evaluate(model, criterion, data_loader, device):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data['image'], data['class']
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

    return top1, top5


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


def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to(TRAIN_ARCH)
    return model


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def train_one_epoch(model, criterion, optimizer, data_loader, device):

    model.train()  #TODO Why model.train() runs 2 times ??? atm

    # Defines statistical variables
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    # Training Loop (Through all data pictures)
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data['image'], data['class']
        inputs = inputs.to(device)
        labels = labels.to(device)

        start_time = time.time()
        inputs, labels = inputs.to(device), labels.to(device)

        # Calculate Network Function (what Network thinks of this image)
        output = model(inputs)
        # Calculate loss
        loss = criterion(output, labels)
        # Resets Gradient to Zeros (clearing it before using in calculations)
        optimizer.zero_grad()
        # Backpropagate loss
        loss.backward()
        # Update the weighs
        optimizer.step()
        # Calculate Accuracy
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        # Update Statistics
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        avgloss.update(loss, inputs.size(0))

    # Print Result for One Epoch of Training
    print('Full train set:  * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {avgloss.avg:.3f}'
          .format(top1=top1, top5=top5, avgloss=avgloss))
    return





class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == '__main__':

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        RandomRotationTransform(angles=[-90, 90, 0, 180, -180]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.75, 1.45), contrast=0.5, saturation=0.5, hue=0.3),
        torchvision.transforms.RandomApply(
            [torchvision.transforms.Grayscale(num_output_channels=3)],
            p=0.35
        ),
        transforms.ToTensor(),
        torchvision.transforms.RandomApply(
            [AddGaussianNoise(0., 1.)],
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
    trainset = Bankset(Bankset.DATASET_PATH, transform_train)
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)

    valset = Bankset(Bankset.VALSET_PATH, transform_val)
    valloader = DataLoader(valset, batch_size=8, shuffle=True, num_workers=0)

    testset = Bankset(Bankset.TESTSET_PATH, transform_test)
    testloader = DataLoader(testset, batch_size=8, shuffle=True, num_workers=0)
    print("Data Loaded")

    # Prepare Model
    original_model = models.resnet18(pretrained=True, progress=True, quantize=False)

    model = create_combined_model(original_model)
    print("Model Created")
    print(model)

    # Create Criterion and Optimilizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[4, 15, 20, 30, 40, 95], gamma=0.8)

    # uncomment to retrain / finetune
    # model.load_state_dict(torch.load(Bankset.BEST_MODEL_PATH), strict=False)
    # print(model)

    best_acc = 0
    model = model.to(torch.device(TRAIN_ARCH))

    # Training Network
    for nEpoch in range(MAX_EPOCH_NUMBER):

        model = model.to(torch.device(TRAIN_ARCH))

        model.train() #TODO Why model.train() runs 2 times ??? atm  (maybe it is enabling calculate gradient mode? - Not really)
        #https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/36
        train_one_epoch(model, criterion, optimizer, trainloader, torch.device(TRAIN_ARCH))

        exp_lr_scheduler.step() # TODO what it does ? atm

        # Evaluate each epoch:
        #TODO why to evaluate every epoch while training? atm
        model.eval() # Not Sure - same as  torch.no_grad(): (disabling calculations of gradient)
        mtop1, mtop5 = evaluate(model, criterion, valloader, torch.device(TRAIN_ARCH))
        if mtop1.avg > best_acc:
            best_acc = mtop1.avg
            best_model = copy.deepcopy(model.state_dict())
            print("Saved ", best_acc)
            torch.save(best_model, Bankset.BEST_MODEL_PATH)

        print('Epoch %d :Evaluation accuracy on all validation images, %2.2f' % (nEpoch, mtop1.avg))

    xtop1, xtop5 = evaluate(model, criterion, testloader, torch.device(TRAIN_ARCH))
    print('Evaluation accuracy on all test images, %2.2f' % (xtop1.avg))
    print("Finished Training")
