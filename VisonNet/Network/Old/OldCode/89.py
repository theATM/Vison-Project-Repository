'''

I tested this file for some time and it is Training! (got 30% so far acc on val) - ATM

'''
from mobilenet_v2 import MobileNetV2, AverageMeter
from PIL import Image
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

#data_path = 'data/imagenet_1k'
saved_model_dir = oldBank.BEST_MODEL_DIR #'quant_mobilenet/'
float_model_file = oldBank.BEST_MODEL_DIR + 'mobilenet_pretrained_float.pth'
scripted_float_model_file = oldBank.BEST_MODEL_DIR + 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = oldBank.BEST_MODEL_DIR + 'mobilenet_quantization_scripted_quantized.pth'


BEST_MODEL_PATH = 'quant_mobilenet_best_model.pth'#'./quant_mobilenet/best_model.pth'


from torch import nn


  
def evaluate(model, criterion, data_loader, device):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():    
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data['image'], data['class']
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            cnt += 1
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', oldBank.os.path.getsize("temp.p")/1e6)
    oldBank.os.remove('temp.p')



def train_one_epoch(model, criterion, optimizer, data_loader, device,i):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    for i, data in enumerate(data_loader, 0):

        inputs, labels = data['image'], data['class']
        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        avgloss.update(loss, inputs.size(0))

    print('Epoch {i:d} Full train set:  * Accuracy {top1.avg:.3f} In Top 5 {top5.avg:.3f} Loss {avgloss.avg:.3f}'
          .format(i=i,top1=top1, top5=top5, avgloss=avgloss))
    return


import random
class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

if __name__ == '__main__':

    transform_train = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(400),
                                          transforms.CenterCrop((400,400)),
                                          #transforms.RandomRotation(180),
                                          MyRotationTransform(angles=[-90, 90, 0, 180, -180]),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          #transforms.RandomPerspective(distortion_scale=0.25, p=0.5, interpolation=3, fill=0),
                                          transforms.ColorJitter(brightness=(0.75,1.25)),
                                          torchvision.transforms.RandomApply(
                                            [torchvision.transforms.Grayscale(num_output_channels=3)],
                                            p=0.25
                                          ),
                                          transforms.ToTensor(),
                                          #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                          transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.5, 2.3), value='random')])
# ~83
    transform_val   = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(400),
                                          transforms.CenterCrop((400,400)),
                                          transforms.ToTensor(),])
                                          #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
    transform_test  = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(400),
                                          transforms.CenterCrop((400,400)),
                                          transforms.ToTensor(),])
                                          #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    trainset = oldBank.Bankset(oldBank.DATASET_PATH, transform_train)
    trainloader = DataLoader(trainset, batch_size=6, shuffle=True, num_workers=4)

    valset = oldBank.Bankset(oldBank.VALSET_PATH, transform_val)
    valloader = DataLoader(valset, batch_size=6, shuffle=True, num_workers=4)

    testset = oldBank.Bankset(oldBank.TESTSET_PATH, transform_test)
    testloader = DataLoader(testset, batch_size=6, shuffle=True, num_workers=4)


    # float_model = load_model(saved_model_dir + float_model_file).to('cpu')

    # print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
    # float_model.eval()

    # # Fuses modules
    # float_model.fuse_model()

    # # Note fusion of Conv+BN+Relu and Conv+Relu
    # print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)


    model = MobileNetV2(num_classes = 7)#models.resnet18(pretrained=True, progress=True, quantize=False)
    #model.layer2 = Identity()
    #model.layer4 = Identity()
    #model.layer4[0].conv1 = nn.Conv2d(128, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #model.layer4[0].downsample[0] = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)

    
    #model.fuse_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 60, 70, 80], gamma=0.1)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    print(model)
    #model.features.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

    #torch.quantization.prepare_qat(model, inplace=True)
    #print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',model.features[1].conv)

    #model.load_state_dict(torch.load("best_quantized.pth"))
    best_acc = 0
    model = model.to(torch.device('cuda'))
    # Train and check accuracy after each epoch
    for nepoch in range(120):
        #model = model.to(torch.device('cuda'))
        model.train()
        train_one_epoch(model, criterion, optimizer, trainloader, torch.device('cuda'),nepoch)
        exp_lr_scheduler.step()
        # if nepoch > 3:
        #     # Freeze quantizer parameters
        #     model.apply(torch.quantization.disable_observer)
        # if nepoch > 2:
        #     # Freeze batch norm mean and variance estimates`
        #     model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        # model = model.to(torch.device('cpu'))
        # # Check the accuracy after each epoch
        # quantized_model = torch.quantization.convert(model.eval(), inplace=False)
        # quantized_model.eval()
        # quantized_model.to(torch.device('cpu'))
        # #qat_model.eval()
        # top1, top5 = evaluate(quantized_model, criterion, testloader, torch.device('cpu'))
        # model = model.to(torch.device('cuda'))
        # Evaluate in some epochs:
        if nepoch % 10 == 0 or nepoch == 199:
            model.eval()
            mtop1, _ = evaluate(model, criterion, valloader, torch.device('cuda'))
            top1, _ = evaluate(model, criterion, testloader, torch.device('cuda'))
            #if(nepoch % 10 and nepoch > 0):
            #    traced_script_module = torch.jit.trace(qat_model, input_batch)
            #    traced_script_module.save(f"quant_mobilenet/mobilenetv2_quantized_{nepoch}.pt")
            if mtop1.avg > best_acc:
                best_acc = mtop1.avg
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, oldBank.BEST_MODEL_DIR + "Epoch" + str(nepoch) + "_" + BEST_MODEL_PATH)

                # input_batch = valset[100]['image'].unsqueeze(0)
                # traced_script_module = torch.jit.trace(quantized_model, input_batch)
                # traced_script_module.save(f"mobilenet_quantized_{top1}.pt")
            print('Epoch %d :Evaluation accuracy on all test images, %2.2f' % (nepoch, top1.avg))
            print('Epoch %d :Evaluation accuracy on all validation images, %2.2f' % (nepoch, mtop1.avg))



    print("Finished Training")
