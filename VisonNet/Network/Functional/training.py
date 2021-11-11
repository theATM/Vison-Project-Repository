import torch
import time
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler

from Network.Bank import bankset as bank
import Network.parameters as par
from Network.Architecture import model as mod
from Network.Functional import evaluate as eva


def main():

    # Load Data
    trainloader, valloader, testloader = bank.loadData()

    # Creating devices  - choosing where will training/eval calculate (gpu or cpu)
    trainDevice = torch.device(par.TRAIN_ARCH)
    evalDevice = torch.device(par.TRAIN_ARCH)

    #Empty GPU Cache before Training starts
    if par.TRAIN_ARCH == 'cuda:0': torch.cuda.empty_cache()

    # Prepare Model
    #used_model = mod.UsedModel(mod.ModelType.Original_Resnet18, arg_pretrained=True,arg_load = par.LOAD_MODEL, arg_load_path="../Models/Original_Resnet18_13-10-2021_07-44_Epoch_0380_Acc_89.71.pth",arg_load_device=par.TRAIN_ARCH)
    used_model = mod.UsedModel(par.USED_MODEL_TYPE)
    used_model.model.to(trainDevice)
    # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
    exp_lr_scheduler = lr_scheduler.MultiStepLR(used_model.optimizer, milestones=[32, 128, 160, 256, 512, 720], gamma=par.SCHEDULER_GAMMA)

    best_acc = 0

    # Training Network
    print('Training Started')
    #training_start_time = time.time()
    for nEpoch in range(par.MAX_EPOCH_NUMBER):
        print('\n' + 'Epoch ' + str(nEpoch+1) + ':')
        # Training Model
        used_model.model.train()
        train_one_epoch(used_model, trainloader, trainDevice, nEpoch)

        # Stepping scheduler
        exp_lr_scheduler.step()

        # Evaluate in some epochs:
        if (nEpoch+1) % par.EVAL_PER_EPOCHS == 0 :
            used_model.model.eval()
            mtop1, mtop5 = eva.evaluate(used_model, valloader, evalDevice)
            # Save If Best
            if mtop1.avg > best_acc:
                best_acc = mtop1.avg
                used_model.saveModel(nEpoch, best_acc)
            if evalDevice == 'cuda:0' : torch.cuda.empty_cache()
            print('Evaluation on epoch %d accuracy on all validation images, %2.2f' % (nEpoch+1, mtop1.avg))

    print('Epoch ' + str(nEpoch+1) + ' completed')

    #Post Training Evaluation
    used_model.model.eval()
    xtop1, _ = eva.evaluate(used_model, testloader, evalDevice)
    print('\n' + 'Evaluation accuracy on all test images, %2.2f' % (xtop1.avg))
    print("Finished Training")

    #Save If Best #TODO is this really comparable to valset?
    if xtop1.avg > best_acc:
        best_acc = mtop1.avg
        used_model.saveModel(par.MAX_EPOCH_NUMBER , best_acc)


def train_one_epoch(used_model, data_loader, trainDevice, nEpoch):

    # Defines statistical variables
    top1 = bank.AverageMeter('Accuracy', ':6.2f')
    top3 = bank.AverageMeter('In Top 3', ':6.2f')
    avgLoss = bank.AverageMeter('Loss', '1.5f')
    epochStartTime = time.time()
    multi_batch_loss = 0.0


    # Training Loop (Through all data pictures)
    for i, data in enumerate(data_loader):
        inputs = torch.autograd.Variable(data['image'].to(trainDevice, non_blocking=True))
        labels = torch.autograd.Variable(data['class'].to(trainDevice, non_blocking=True))

        # passes and weights update
        with torch.set_grad_enabled(True):

            # Calculate Network Function (what Network thinks of this image)
            output = used_model.model(inputs)
            # Calculate loss
            loss = used_model.criterion(output, labels)
            # Backpropagate loss
            loss.backward()
            multi_batch_loss += loss;

            # Calculate, minding gradient batch accumulation
            if ((i+1) % par.GRAD_PER_BATCH) == 0:
                # Normalize loss to account for batch accumulation
                multi_batch_loss = multi_batch_loss / par.GRAD_PER_BATCH
                #Clipping the gradient
                clip_grad_norm_(used_model.model.parameters(), max_norm = 1)
                # Update the weighs
                used_model.optimizer.step()
                # Resets Gradient to Zeros (clearing it before using it next time in calculations)
                used_model.optimizer.zero_grad()


                if (i+1)  % (par.GRAD_PER_BATCH * 128) == 0 :
                    print('Image ' + str(i+1) + ' Current Loss {loss:.3f}'
                          .format(loss=multi_batch_loss.item()))

                multi_batch_loss = 0.0

            # Calculate Accuracy
            acc1, acc3 = eva.accuracy(output, labels, topk=(1, 3))
            # Update Statistics
            top1.update(acc1[0], inputs.size(0))
            top3.update(acc3[0], inputs.size(0))
            avgLoss.update(loss, inputs.size(0))

    # Print Result for One Epoch of Training
    print('Epoch ' + str(nEpoch+1) +':  * Accuracy {top1.avg:.3f} | In Top 3 {top3.avg:.3f} | Loss {avgLoss.avg:.3f} | Used Time {epochTime:.2f} s'
          .format(top1=top1, top3=top3, avgLoss=avgLoss,epochTime = time.time() - epochStartTime))


if __name__ == '__main__':
    #Run main function
    main()


