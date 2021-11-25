import torch
import enum
from torchvision import transforms

import Network.Bank.bankset as bank
import Network.Bank.transforms as trans
import Network.Bank.banksethelpers as bah
import Network.parameters as par
import Network.Architecture.model as mod
import Network.Architecture.modeltype as modtype
from skimage import io

MODEL_PATH = '../../Models/Quantout/ResQuant84or96First.pt' #"../../Models/Quantin/Original_Resnet18_13-10-2021_07-44_Epoch_0380_Acc_89.71.pthEpoch_0240_Acc_93.68_rs.pth"
#'../Models/OrgResnet18_11-09-2021_17-24_Epoch_0060_Acc_17.62.pth' #'../Models/resnetTa94pretrained.pth'
MODEL_TYPE = modtype.ModelType.Original_Resnet18


class EvalMode(enum.Enum):
    EVAL = 1
    TEST = 2


MODE = EvalMode.EVAL

Image_PATH = '../../../Data/testset/500/63.jpg'
Image_Label = 500


def __evaluateMain():
    #Load Data
    _, _, testloader = bank.loadData(arg_load_train=False,arg_load_val=False,arg_load_test=True)
    evalDevice = torch.device(par.QUANT_DEVICE)#par.TRAIN_ARCH)
    #Empty GPU Cache before Evaluation starts
    if par.TRAIN_ARCH == 'cuda:0': torch.cuda.empty_cache()
    #used_model = mod.UsedModel(MODEL_TYPE, arg_load_path=MODEL_PATH, arg_load=True, arg_load_device=evalDevice)
    used_model = mod.UsedModel(modtype.ModelType.Original_Resnet18, arg_load_path=MODEL_PATH)
    used_model.model = torch.jit.load(MODEL_PATH)
    print("Evaluation Started")
    used_model.model.to(evalDevice)
    used_model.model.eval()
    model_accuracy, _ = evaluate(used_model, testloader, evalDevice)
    print('Evaluation Accuracy on all test images, %2.2f' % (model_accuracy.avg))
    print("Evaluation Finished")



def evaluate(used_model, data_loader, device):

    top1 = bah.AverageMeter('Accuracy', ':6.2f')
    top3 = bah.AverageMeter('In Top 3', ':6.2f')

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = data['image'].to(device)
            labels = data['class'].to(device)
            output = used_model.model(inputs)
            loss = used_model.criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top3.update(acc5[0], inputs.size(0))

    return top1, top3



def __testMain():

    test_device = torch.device(par.TRAIN_ARCH)
    used_model = mod.UsedModel('Original_Resnet18', arg_load_path=MODEL_PATH, arg_load=True)
    #used_model.model.to(test_device)
    used_model.model.eval()

    transform_test = trans.TRANSFORM_BLANK

    image = io.imread(Image_PATH)
    image = transform_test(image)
    image = image[None,:,:,:]
    #image = image.to(test_device)

    with torch.no_grad():
        output = used_model.model(image)
        print(output)
        print("classes ",end="")
        print(bank.classes)
        pred_val, pred = output.topk(1, 1, True, True)
        pred_val = pred_val.numpy()
        print("Choosen nr "+ str(pred) + " With sertanty = " + str(pred_val))
        suggest = bank.anticlasses.get(pred.item())
        suggest_val = pred_val[0][0]
        correct = Image_Label
        correct_val = output.numpy()[0][0][bank.anticlasses.get(correct)][0]
        hit = pred.eq(bank.classes.get(str(Image_Label)))

        print("Image " + str(Image_PATH) + " recognised as " + str(suggest) + " zl, with " + str(suggest_val) + " certainty")


        if hit is True:
            print("This is Correct")
        else:
            print("This is Not Correct")

        print("This image should be " + "recognised as " + str(correct) + " zl, " + "( " + str(correct_val) + " )")



def multi():
    _,_,testloader = bank.loadData(arg_load_train=False, arg_load_val=False, arg_load_test=True)
    test_device = torch.device(par.TRAIN_ARCH)
    # Empty GPU Cache before Testing starts
    if par.TRAIN_ARCH == 'cuda:0': torch.cuda.empty_cache()
    used_model = mod.UsedModel('Original_Resnet18', arg_load_path=MODEL_PATH, arg_load=True)
    used_model.model.eval()
    singleBatch = next(iter(testloader))
    input, label, name = singleBatch['image'], singleBatch['class'], singleBatch['name']

    with torch.no_grad():
        output = used_model.model(input)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.contiguous().view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        print(str(correct_k))


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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    # Run main function
    if(MODE == EvalMode.EVAL):
        __evaluateMain()
    elif(MODE == EvalMode.TEST):
        __testMain()
    else:
        print('Unknown mode set')