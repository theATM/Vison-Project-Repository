# Here you can insert bare model .pt and pack it with some useful info and save it as .pth
from Network.Architecture import model as mod
from Network.Architecture import modeltype as modtype
import Network.parameters as par
#NOT SAFE TODO
model_type = modtype.ModelType.Original_Resnet18
model_path = '../../Models/Original_Resnet18_11-10-2021_21-21_Epoch_0020_Acc_21.95.pth'
model_epoch = 0
model_accuracy = 0.36
model_loss = 0.1

used_model = mod.UsedModel(model_type, arg_load_raw=True, arg_load_path=model_path)


used_model.saveModel(arg_epoch=model_epoch, arg_acc=model_accuracy, arg_loss=model_loss)

