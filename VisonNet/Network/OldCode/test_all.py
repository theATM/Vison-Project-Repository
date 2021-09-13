'''

This file to best of mine knowledge is proto quantize.py - ATM
Will Work only on Linux with CPU

'''
from mobilenet_v2 import MobileNetV2, AverageMeter
#from PIL import Image
import torch
from torchvision import transforms
import torchvision
from torch import nn

import torch.optim as optim
from torch.utils.data import DataLoader
import bankset_old as oldBank
import copy
import torchvision.models.quantization as models

from torch.optim import lr_scheduler
import time



BEST_MODEL_PATH = oldBank.BEST_MODEL_DIR + 'quant_mobilenet_best_model.pth'


from torch import nn

def create_combined_model(model_fe):
  model_fe_features = nn.Sequential(
    model_fe.conv1,
    model_fe.bn1,
    model_fe.relu,
    model_fe.maxpool,
    model_fe.layer1,
    model_fe.layer2,
    model_fe.layer3,
    #model_fe.layer4,# without this
    model_fe.avgpool,
  )

  new_head = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(256, 512),
    nn.Linear(512, 7),
  )

  new_model = nn.Sequential(
    model_fe_features,
    nn.Flatten(1),
    new_head,
  )
  return new_model



def add_quant_stubs(model_fe):
  new_model = nn.Sequential(
    torch.quantization.QuantStub(),
    model_fe[0][0],
    model_fe[0][1],
    model_fe[0][2],
    model_fe[0][3],
    nn.Sequential(
        model_fe[0][4][0],
        model_fe[0][4][1],
    ),
    nn.Sequential(
        model_fe[0][5][0],
        model_fe[0][5][1],
    ),
    nn.Sequential(
        model_fe[0][6][0],
        model_fe[0][6][1],
    ),
    model_fe[0][7],
    model_fe[1],
    torch.quantization.DeQuantStub(),
    model_fe[2],
  )

  return new_model
  
def evaluate(model, criterion, data_loader, device):
    model.eval()
    confusion_matrix = torch.zeros(7,7)
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():    
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data['image'], data['class']
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            _, preds = torch.max(output, 1)
            for i, p in enumerate(preds):
                confusion_matrix[labels[i]][p] += 1
            loss = criterion(output, labels)

            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

    print(confusion_matrix)
    return top1, top5


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    transform_test  = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(224),
                                          transforms.CenterCrop((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703]),])

    testset = oldBank.Bankset(oldBank.TESTSET_PATH, transform_test)
    testloader = DataLoader(testset, batch_size=6, shuffle=True, num_workers=4)

    trainset = oldBank.Bankset(oldBank.DATASET_PATH, transform_test)
    trainloader = DataLoader(trainset, batch_size=6, shuffle=True, num_workers=4)

    original_model = models.resnet18(pretrained=True, progress=True, quantize=False)
    model = original_model
    #model = create_combined_model_orig(original_model)
    model = create_combined_model(model)
    print(model)
    for name, param in model.named_parameters():
        print(name)
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    #exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[35, 50, 90], gamma=0.1)#


    model.load_state_dict(torch.load("best_quantized.pth"))

    best_acc = 0
    num_train_batches = 8
    model = model.to(torch.device('cuda'))
    model.eval()
    #top1, top5 = evaluate(model, criterion, testloader, torch.device('cuda'))

    #print('Evaluation accuracy on all test images, %2.2f'%(top1.avg))



    model = model.to(torch.device('cpu'))
    modules_to_fuse = [['1', '2'],
                   ['5.0.conv1', '5.0.bn1'],
                   ['5.0.conv2', '5.0.bn2'],
                   ['5.1.conv1', '5.1.bn1'],
                   ['5.1.conv2', '5.1.bn2'],

                   ['6.0.conv1', '6.0.bn1'],
                   ['6.0.conv2', '6.0.bn2'],
                   ['6.0.downsample.0', '6.0.downsample.1'],
                   ['6.1.conv1', '6.1.bn1'],
                   ['6.1.conv2', '6.1.bn2'],

                   ['7.0.conv1', '7.0.bn1'],
                   ['7.0.conv2', '7.0.bn2'],
                   ['7.0.downsample.0', '7.0.downsample.1'],
                   ['7.1.conv1', '7.1.bn1'],
                   ['7.1.conv2', '7.1.bn2']]
    
    model = add_quant_stubs(model)
    print(model)
    model = torch.quantization.fuse_modules(model, modules_to_fuse)
    
    print(model)
    torch.backends.quantized.engine = 'qnnpack' #THIS WORKS ONLY ON LINUX and ON CPU!!!(Otherwise Error)

    white_list = torch.quantization.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST
    white_list.remove(torch.nn.modules.linear.Linear)
    qconfig_dict = dict()
    for e in white_list:
        qconfig_dict[e] = torch.quantization.get_default_qconfig('qnnpack')

    torch.quantization.propagate_qconfig_(model, qconfig_dict=qconfig_dict)
    #model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    print(model.qconfig)





    torch.quantization.prepare(model, inplace=True)
    model.eval()

    print(model)
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data['image'], data['class']
            model(inputs)

    torch.quantization.convert(model, inplace=True)

    #model = add_quant_stubs(model)
    print(model)



    best_acc = 0
    num_train_batches = 8
    model = model.to(torch.device('cpu'))
    model.eval()
    top1, top5 = evaluate(model, criterion, testloader, torch.device('cpu'))

    print('Evaluation accuracy on all test images, %2.2f'%(top1.avg))

    for i, data in enumerate(testloader, 0):
        inputs, labels = data['image'], data['class']
        traced_script_module = torch.jit.trace(model, inputs)
        traced_script_module.save("rn18quantized_94_last.pt")
        break
#                    ['conv1', 'layer1.0.bn1'],
#                    ['layer1.0.conv2', 'layer1.0.bn2'],
#                    ['layer1.1.conv1', 'layer1.1.bn1'],
#                    ['layer1.1.conv2', 'layer1.1.bn2'],
#                    ['layer2.0.conv1', 'layer2.0.bn1'],
#                    ['layer2.0.conv2', 'layer2.0.bn2'],
#                    ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
#                    ['layer2.1.conv1', 'layer2.1.bn1'],
#                    ['layer2.1.conv2', 'layer2.1.bn2'],
#                    ['layer3.0.conv1', 'layer3.0.bn1'],
#                    ['layer3.0.conv2', 'layer3.0.bn2'],
#                    ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
#                    ['layer3.1.conv1', 'layer3.1.bn1'],
#                    ['layer3.1.conv2', 'layer3.1.bn2'],
#                    ['layer4.0.conv1', 'layer4.0.bn1'],
#                    ['layer4.0.conv2', 'layer4.0.bn2'],
#                    ['layer4.0.downsample.0', 'layer4.0.downsample.1'],
#                    ['layer4.1.conv1', 'layer4.1.bn1'],
#                    ['layer4.1.conv2', 'layer4.1.bn2']]