import torch
import enum
from torchvision import transforms

import Network.Bank.bankset as bank
import Network.Bank.banksethelpers as bah
import Network.parameters as par
import Network.Architecture.model as mod


def main():
    #Load Data
    _, testloader,_  = bank.loadData(arg_load_train=False,arg_load_val=True,arg_load_test=False)

    #Define Eval Device
    eval_device = par.QUANT_DEVICE if par.EVAL_LOAD_MODEL_IS_QUANTIZED else par.TRAIN_DEVICE

    #Empty GPU Cache before Evaluation starts
    if eval_device == 'cuda:0' : torch.cuda.empty_cache()

    #Load Model
    used_model = mod.UsedModel(par.MODEL_USED_MODEL_TYPE, arg_load_path=par.MODEL_LOAD_MODEL_PATH, arg_load=True,
                               arg_load_device=eval_device,  arg_load_quantized=par.EVAL_LOAD_MODEL_IS_QUANTIZED)
    used_model.model.to(eval_device)

    #Eval
    print("\nEvaluation Started")
    print("Using Testset")
    used_model.model.eval()
    model_accuracy, _,_ = evaluate(used_model, testloader, eval_device)
    print('Evaluation Accuracy on all test images, %2.2f' % (model_accuracy.avg))
    print("Evaluation Finished")



def evaluate(used_model, data_loader, device):

    top1 = bah.AverageMeter('Accuracy', ':6.2f')
    top3 = bah.AverageMeter('In Top 3', ':6.2f')
    avg_loss = bah.AverageMeter('Loss', ':6.2f')

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = data['image'].to(device)
            labels = data['class'].to(device)
            output = used_model.model(inputs)
            loss = used_model.criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top3.update(acc5[0], inputs.size(0))
            avg_loss.update(loss,inputs.size(0))

    return top1, top3, avg_loss


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
    main()