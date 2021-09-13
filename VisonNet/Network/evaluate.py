import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import bankset as bank
import parameters as par
import model as mod
import training as train

MODEL_PATH = ''
MODE = 'Evaluate'


def __evaluateMain():
    #Load Data
    _, _, testloader = bank.loadData(arg_load_train=False,arg_load_val=False,arg_load_test=True)
    evalDevice = torch.device(par.TRAIN_ARCH)
    #Empty GPU Cache before Evaluation starts
    if par.TRAIN_ARCH == 'cuda:0': torch.cuda.empty_cache()
    used_model = mod.UsedModel('Original_Resnet18', loadPath=MODEL_PATH,load=True)
    print("Evaluation Started")
    used_model.model.eval()
    model_accuracy, _ = evaluate(used_model, testloader, evalDevice)
    print('Evaluation Accuracy on all test images, %2.2f' % (model_accuracy.avg))
    print("Evaluation Finished")



def evaluate(used_model, data_loader, device):

    top1 = bank.AverageMeter('Accuracy', ':6.2f')
    top3 = bank.AverageMeter('In Top 3', ':6.2f')

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

    _, _, testloader = bank.loadData(arg_load_train=False, arg_load_val=False, arg_load_test=True)
    test_device = torch.device(par.TRAIN_ARCH)
    #Empty GPU Cache before Testing starts
    if par.TRAIN_ARCH == 'cuda:0': torch.cuda.empty_cache()
    used_model = mod.UsedModel('Original_Resnet18', loadPath=MODEL_PATH, load=True)




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
    # Run main function
    if(MODE == 'Eval'):
        __evaluateMain()
    elif(MODE == 'Test'):
        __testMain()
    else:
        print('Unknown mode set')