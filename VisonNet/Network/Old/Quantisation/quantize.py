import torch
import torch.optim as optim

from torchvision import transforms
from torch import nn

import Network.parameters as par
import Network.Bank.bankset as bank
import Network.Architecture.model as mod





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
    # If you have gpu On Linux go for it - not sure if it works on gpu though
    quantDevice = par.QUANT_DEVICE #torch.device('cpu')

    transform_test = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(224),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701],
                                                              std=[0.24467267, 0.23742135, 0.24701703]), ])

    #Load Data
    #testset = bank.Bankset("/VisonApp/testset", transform_test)
    #testloader = DataLoader(testset, batch_size=2, shuffle=True, num_workers=0)

    #trainset = bank.Bankset("/VisonApp/dataset", transform_test)
    #trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)

    trainloader, _, testloader = bank.loadData(arg_load_train=True, arg_load_val=False, arg_load_test=True,
                                         arg_trans_train=transform_test, quantisation_mode=True)


    #Load Original Model
    #original_model = models.resnet18(pretrained=True, progress=True, quantize=False)


    #Load Our Best Model
    model = mod.UsedModel(mod.ModelType.Original_Resnet18, arg_load=True, arg_load_path= par.QUANT_MODEL_PATH, arg_load_device=par.QUANT_DEVICE, arg_load_raw=True)
    model.optimizer = optim.Adam(model.model.parameters(), lr=par.INITIAl_LEARNING_RATE)
    #criterion = nn.CrossEntropyLoss()
    print('Loaded trained model')



    #Evaluate Our Model
    best_acc = 0
    num_train_batches = 8
    model.model.to(quantDevice)
    model.model.eval()

    if DO_EVALUATE:
        top1, top5 = evaluate(model.model, model.criterion, testloader, quantDevice)
        print('Evaluation accuracy on all test images, %2.2f'%(top1.avg))


    model.model.eval()
    model.addQuantStubs()


    #print(model)
    model.model.eval()
    model.model = torch.quantization.fuse_modules(model.model, model.getLayersToFuse())
    #print(model)





    print('\nQuantization Started')
    if BACKEND_ENGINE == 'qnnpack':
        # qnnpack - works for ARM (Linux)
        print("Using qnnpack backend engine")
        torch.backends.quantized.engine = 'qnnpack'  # atm - not working in windows 10

        #Preform Static Quantisation:

        white_list = torch.quantization.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST
        #white_list = torch.quantization.get_default_qconfig_propagation_list()
        #xx = torch.quantization.get_default_qat_module_mappings()
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
                if (i + 1) % 2 == 0: break
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
        top1, top5 = evaluate(model.model, testloader, quantDevice)

        print('Evaluation accuracy on all test images, %2.2f' % (top1.avg))

    # save for mobile
    model.model.to("cpu")
    for i, data in enumerate(testloader):
        inputs, labels = data['image'], data['class']
        traced_script_module = torch.jit.trace(model.model, inputs)
        traced_script_module.save("rn18quantized.pt")
        break

    print('Model Saved Successfully')

