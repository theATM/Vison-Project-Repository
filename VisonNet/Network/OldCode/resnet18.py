import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import bankset_old as oldBank
import copy

FINAL_MODEL_PATH = oldBank.BEST_MODEL_DIR + 'resnet_final_model.pth'
BEST_MODEL_PATH = oldBank.BEST_MODEL_DIR + 'resnet_best_model.pth'
EPOCHS = 1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, optimizer, criterion):
    model.to(device)

    running_loss = 0.0
    running_corrects = 0

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data['image'], data['class']
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, optimizer, criterion):
    model.to(device)

    running_loss = 0.0
    running_corrects = 0

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data['image'], data['class']
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc

def test(model, dataloader):
    model.to(device)

    running_corrects = 0
    for i, data in enumerate(dataloader, 0):
            inputs, labels = data['image'], data['class']
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / len(dataloader.dataset)
    return acc

if __name__ == "__main__":

    transform_train = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(400),
                                          transforms.CenterCrop((400,400)),
                                          transforms.RandomRotation(180),
                                          transforms.ColorJitter(brightness=(1.0,1.2)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    transform_val   = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(400),
                                          transforms.CenterCrop((400,400)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
    transform_test  = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(400),
                                          transforms.CenterCrop((400,400)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    trainset = oldBank.Bankset(oldBank.DATASET_PATH, transform_train)
    trainloader = DataLoader(trainset, batch_size=6, shuffle=True, num_workers=4)

    valset = oldBank.Bankset(oldBank.VALSET_PATH, transform_val)
    valloader = DataLoader(valset, batch_size=6, shuffle=True, num_workers=4)

    testset = oldBank.Bankset(oldBank.TESTSET_PATH, transform_test)
    testloader = DataLoader(testset, batch_size=6, shuffle=True, num_workers=4)

    classess = ('10', '20', '50', '100', '200', '500', 'none')

    #Define model
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    #model.load_state_dict(torch.load(PATH_LOAD))
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = None
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()

        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion)
        exp_lr_scheduler.step()
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))

        model.eval()
        val_loss, val_acc = validate_epoch(model, valloader, optimizer, criterion)

        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, BEST_MODEL_PATH)

    print('Best val Acc: {:4f}'.format(best_acc))
    print('Finished Training')

    if(best_model):
        model.load_state_dict(best_model)

    model.eval()
    model.to(device)
    acc = test(model, trainloader)
    print('Test Acc: {:.4f}'.format(acc))

    torch.save(model.state_dict(), FINAL_MODEL_PATH)