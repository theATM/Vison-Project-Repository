import torch
import torchvision
import torch.optim as optim
import copy
import torchvision.models.quantization as models

from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


import Network.parameters as par
import Network.bankset as bank
import Network.model as mod





DO_EVALUATE = False


def create_combined_model(model_fe):
    model_fe_features = nn.Sequential(
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        # model_fe.layer4,# without this
        model_fe.avgpool,
    )

    new_head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(256, 7),
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
    confusion_matrix = torch.zeros(7, 7)
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

    # Choose quantization engine
    BACKEND_ENGINE = ''
    if 'qnnpack' in torch.backends.quantized.supported_engines:
        # This Engine Works ONLY on Linux
        # We will use it
        BACKEND_ENGINE = 'qnnpack'
    elif 'fbgemm' in torch.backends.quantized.supported_engines:
        # This Engine works on Windows (and Linux?)
        # We won't be using it
        BACKEND_ENGINE = 'fbgemm'
        print("FBGEMM Backend Engine is not supported")
        exit(-2)
    else:
        BACKEND_ENGINE = 'none'
        print("No Proper Backend Engine found")
        exit(-3)

    # Choose quantization device (cpu/gpu)
    # If you have gpu On Linux go for it
    quantDevice = torch.device('cpu')

    transform_test = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(224),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701],
                                                              std=[0.24467267, 0.23742135, 0.24701703]), ])

    #Load Data
    testset = bank.Bankset(par.TESTSET_PATH, transform_test)
    testloader = DataLoader(testset, batch_size=6, shuffle=True, num_workers=4)

    trainset = bank.Bankset(par.DATASET_PATH, transform_test)
    trainloader = DataLoader(trainset, batch_size=6, shuffle=True, num_workers=4)

    #Load Original Model
    original_model = models.resnet18(pretrained=True, progress=True, quantize=False)


    #Load Our Best Model
    model = mod.UsedModel('Original_Resnet18',load = True, loadPath = par.BEST_MODEL_PATH)
    criterion = nn.CrossEntropyLoss()
    print('Loaded trained model')



    #Evaluate Our Model
    best_acc = 0
    num_train_batches = 8
    model.model.to(quantDevice)
    model.model.eval()

    if DO_EVALUATE:
        top1, top5 = evaluate(model.model, criterion, testloader, quantDevice)
        print('Evaluation accuracy on all test images, %2.2f'%(top1.avg))

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
    model.model.eval()
    model.model = add_quant_stubs(model.model)
    #print(model)
    model.model.eval()
    model.model = torch.quantization.fuse_modules(model.model, modules_to_fuse)
    #print(model)





    print('\nQuantization Started')
    if BACKEND_ENGINE == 'qnnpack':
        # qnnpack - works for ARM (Linux)
        print("Using qnnpack backend engine")
        torch.backends.quantized.engine = 'qnnpack'  # atm - not working in windows 10

        #Preform Static Quantisation:

        #white_list = torch.quantization.DEFAULT_QCONFIG_PROPAGATE_ALLOW_LIST
        white_list = torch.quantization.get_default_qconfig_propagation_list()
        xx = torch.quantization.get_default_qat_module_mappings()
        white_list.remove(torch.nn.modules.linear.Linear)
        qconfig_dict = dict()
        model.model.eval()
        for e in white_list:
            qconfig_dict[e] = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.propagate_qconfig_(model.model, qconfig_dict)
        model.model.eval()
        torch.quantization.prepare(model.model, inplace=True)
        model.model.eval()
        print("\nStarting Quantizising Imputs")
        with torch.no_grad():
            for i, data in enumerate(trainloader, 0):
                if i % 1000 == 0 : print("Progress = " , i)
                inputs, labels = data['image'], data['class']
                model.model(inputs)
        print("Imputs Quantized")
        #v = model.model(trainloader.dataset.images[0]['class'])

        torch.quantization.convert(model.model, inplace=True)
        print("Model Quantized")


    elif BACKEND_ENGINE == 'fbgemm':
        # fbgemm - works in x86 machines (Windows)
        print("Using fbgemm backend engine is not supported")
        exit(-2)
    else:
        print("Using unknown backend engine - aborting")
        exit(-3)

    model.model.eval()
    if DO_EVALUATE:
        best_acc = 0
        num_train_batches = 4
        model.model.to(quantDevice)
        model.model.eval()
        top1, top5 = evaluate(model.model, criterion, testloader, quantDevice)

        print('Evaluation accuracy on all test images, %2.2f' % (top1.avg))

    # save for mobile
    model.model.to("cuda")
    for i, data in enumerate(testloader):
        inputs, labels = data['image'], data['class']
        traced_script_module = torch.jit.trace(model.model, inputs)
        traced_script_module.save("rn18quantized.pt")
        break

    print('Model Saved Successfully')

