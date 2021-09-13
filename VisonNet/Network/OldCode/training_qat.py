'''

This file looks like training olready quanticized??? - ATM


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
float_model_file = oldBank.BEST_MODEL_DIR +'protoTE_mobilenet_pretrained_float.pth'
scripted_float_model_file = oldBank.BEST_MODEL_DIR +'protoTE_mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = oldBank.BEST_MODEL_DIR + 'protoTE_mobilenet_quantization_scripted_quantized.pth'


BEST_MODEL_PATH = 'protoTE_best_model.pth'


from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
            #nn.ReLU6(inplace=False)
        )


def create_combined_model(model_fe):
  print("xdxd\n\n\nxdxdxd\nn\n")
  # Step 1. Isolate the feature extractor.\
  
  #final_layer = ConvBNReLU(256, 512, kernel_size=3, stride=2)
  model_fe_features = nn.Sequential(
    #model_fe.quant,  # Quantize the input
    model_fe.conv1,
    model_fe.bn1,
    model_fe.relu,
    model_fe.maxpool,
    model_fe.layer1,
    model_fe.layer2,
    model_fe.layer3,
    #model_fe.layer4, #without this
    model_fe.avgpool,
    #model_fe.dequant,  # Dequantize the output
  )

  # Step 2. Create a new "head"
  new_head = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(256, 512),
    nn.Linear(512, 7),
  )

  # Step 3. Combine, and don't forget the quant stubs.
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
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
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')



def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for i, data in enumerate(data_loader, 0):

        inputs, labels = data['image'], data['class']
        inputs = inputs.to(device)
        labels = labels.to(device)

        start_time = time.time()
        cnt += 1
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

        # if cnt >= ntrain_batches:
        #     print('Loss', avgloss.avg)

        #     print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #           .format(top1=top1, top5=top5))
        #     return

    print('Full train set:  * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {avgloss.avg:.3f}'
          .format(top1=top1, top5=top5, avgloss=avgloss))
    return


import random
class MyRotationTransform:
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

    transform_train = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(224),
                                          transforms.CenterCrop((224,224)),
                                          #transforms.RandomRotation(180),
                                          MyRotationTransform(angles=[-90, 90, 0, 180, -180]),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.RandomPerspective(distortion_scale=0.35, p=0.5, interpolation=3, fill=0),
                                          transforms.ColorJitter(brightness=(0.75,1.8), contrast=0.5, saturation=0.3, hue=0.25),
                                          torchvision.transforms.RandomApply(
                                            [torchvision.transforms.Grayscale(num_output_channels=3)],
                                            p=0.35#0.6
                                          ),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703]),
                                          transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.5, 2.3), value='random'),
                                          torchvision.transforms.RandomApply(
                                            [AddGaussianNoise(0., 1.)],
                                            p=0.35#0.75
                                          )])

    transform_val   = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(224),
                                          transforms.CenterCrop((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703])])
        
    transform_test  = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(224),
                                          transforms.CenterCrop((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703])])

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


    original_model = models.resnet18(pretrained=True, progress=True, quantize=False)
    #model.layer2 = Identity()
    #model.layer4 = Identity()
    #model.layer4[0].conv1 = nn.Conv2d(128, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #model.layer4[0].downsample[0] = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)

    
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(256, 7)
    #original_model.fuse_model()
    model = original_model
    model = create_combined_model(original_model)
    print(model)
    #qat_model = torch.load("quant_mobilenet/best_float.pth")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 45, 55], gamma=0.1)
    #exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    #model[0].qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

    #torch.quantization.prepare_qat(model, inplace=True)
    #print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',model.features[1].conv)

    model.load_state_dict(torch.load("best_quantized.pth"))
    print(model)
    # return
    best_acc = 0
    num_train_batches = 8
    model = model.to(torch.device('cuda'))
    # Train and check accuracy after each epoch
    for nepoch in range(80):
        model = model.to(torch.device('cuda'))
        model.train()
        train_one_epoch(model, criterion, optimizer, trainloader, torch.device('cuda'))
        exp_lr_scheduler.step()
        # if nepoch > 3:
        #     # Freeze quantizer parameters
        #     model.apply(torch.quantization.disable_observer)
        # if nepoch > 2:
        #     # Freeze batch norm mean and variance estimates
        #     model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        # model = model.to(torch.device('cpu'))
        # # Check the accuracy after each epoch
        # quantized_model = torch.quantization.convert(model.eval(), inplace=False)
        # quantized_model.eval()
        # quantized_model.to(torch.device('cpu'))
        #qat_model.eval()
        # top1, top5 = evaluate(quantized_model, criterion, testloader, torch.device('cpu'))
        # model = model.to(torch.device('cuda'))
        model.eval()
        mtop1, mtop5 = evaluate(model, criterion, valloader, torch.device('cuda'))
        xtop1, xtop5 = evaluate(model, criterion, testloader, torch.device('cuda'))
        #if(nepoch % 10 and nepoch > 0):
        #    traced_script_module = torch.jit.trace(qat_model, input_batch)
        #    traced_script_module.save(f"quant_mobilenet/mobilenetv2_quantized_{nepoch}.pt")
        if xtop1.avg > best_acc:
            best_acc = xtop1.avg
            best_model = copy.deepcopy(model.state_dict())
            print("Saved ", best_acc)
            torch.save(best_model, "best_quantized.pth")
            # input_batch = valset[100]['image'].unsqueeze(0)
            # traced_script_module = torch.jit.trace(quantized_model, input_batch)
            # traced_script_module.save(f"resnet_quantized_{mtop1}.pt")

            #input_batch = valset[100]['image'].unsqueeze(0)
            #traced_script_module = torch.jit.trace(quantized_model, input_batch)
            #traced_script_module.save("mobilenetv2_quantized.pt")
            #torch.save(best_model, BEST_MODEL_PATH)
        print('Epoch %d :Evaluation accuracy on all test images, %2.2f'%(nepoch, xtop1.avg))
        print('Epoch %d :Evaluation accuracy on all validation images, %2.2f'%(nepoch, mtop1.avg))
    print("Finished Training")
