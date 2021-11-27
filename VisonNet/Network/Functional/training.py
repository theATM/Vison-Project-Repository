import torch
import time
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler

from Network.Bank import bankset as bank
import Network.Bank.banksethelpers as bah
import Network.parameters as par
from Network.Architecture import model as mod
from Network.Functional import evaluate as eva


def main():

    # Load Data
    trainloader, valloader, testloader = bank.loadData(single_batch_test=par.DATA_SINGLE_BATCH_TEST_ENABLE)

    # Creating devices  - choosing where will training/eval calculate (gpu or cpu)
    trainDevice = torch.device(par.TRAIN_DEVICE)
    evalDevice = torch.device(par.TRAIN_DEVICE)

    #Empty GPU Cache before Training starts
    if par.TRAIN_DEVICE == 'cuda:0': torch.cuda.empty_cache()

    # Prepare Model
    if par.TRAIN_LOAD_MODEL_ENABLE is True:
        used_model = mod.UsedModel(par.MODEL_USED_MODEL_TYPE, arg_load = par.TRAIN_LOAD_MODEL_ENABLE, arg_load_path=par.MODEL_LOAD_MODEL_PATH, arg_load_device=par.TRAIN_DEVICE)
        used_model.model.to(trainDevice)

    else:
        used_model = mod.UsedModel(par.MODEL_USED_MODEL_TYPE, arg_pretrained=True)
        used_model.model.to(trainDevice)
        #used_model.optimizer.to(trainDevice)


    # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
    exp_lr_scheduler = lr_scheduler.MultiStepLR(used_model.optimizer, milestones=[32, 128, 160, 256, 512, 720], gamma=par.TRAIN_SCHEDULER_GAMMA)

    best_acc = 0
    # Training Network
    print('Training Started')
    #training_start_time = time.time()
    for nEpoch in range(used_model.start_epoch, par.TRAIN_MAX_EPOCH_NUMBER):
        print('\n' + 'Epoch ' + str(nEpoch+1) + ':')
        # Training Model
        used_model.model.train()
        train_one_epoch(used_model, trainloader, trainDevice, nEpoch)

        # Stepping scheduler
        exp_lr_scheduler.step()

        # Evaluate in some epochs:
        if (nEpoch+1) % par.TRAIN_EVAL_PER_EPOCHS == 0 :
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
        used_model.saveModel(par.TRAIN_MAX_EPOCH_NUMBER, best_acc)


def train_one_epoch(used_model, data_loader, trainDevice, nEpoch):

    # Defines statistical variables
    top1 = bah.AverageMeter('Accuracy', ':6.2f')
    top3 = bah.AverageMeter('In Top 3', ':6.2f')
    avgLoss = bah.AverageMeter('Loss', '1.5f')
    epochStartTime = time.time()
    multi_batch_loss = 0.0

    # Training Loop (Through all data pictures)

    for i, data in enumerate(data_loader):
        #try:
        inputs = torch.autograd.Variable(data['image'].to(trainDevice, non_blocking=True))
        labels = torch.autograd.Variable(data['class'].to(trainDevice, non_blocking=True))
        #except (IOError, ValueError) as e:
        #    print('could not read the file ', e, 'hence skipping it.')
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
            if ((i+1) % par.TRAIN_GRAD_PER_BATCH) == 0:
                # Normalize loss to account for batch accumulation
                multi_batch_loss = multi_batch_loss / par.TRAIN_GRAD_PER_BATCH
                #Clipping the gradient
                clip_grad_norm_(used_model.model.parameters(), max_norm = 1)
                # Update the weighs
                used_model.optimizer.step()
                # Resets Gradient to Zeros (clearing it before using it next time in calculations)
                used_model.optimizer.zero_grad()


                if (i+1)  % (par.TRAIN_GRAD_PER_BATCH * 128) == 0 :
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


