# Here you can insert bare model .pt and pack it with some useful info and save it as .pth
import copy
import torch.optim as optim
import torch

from Network.Architecture import model as mod
from Network.Architecture import modeltype as modtype
import Network.parameters as par
#NOT SAFE TODO

'''Bare bone tool to resave models - atm'''


model_type = modtype.ModelType.Original_Resnet18
model_path = par.PATH_PREFIX + '/' + 'Models/Quantin/Original_Resnet18_13-10-2021_07-44_Epoch_0380_Acc_89.71.pthEpoch_0240_Acc_93.68.pth'
model_save_path = model_path.rsplit('.',1)[0] + '_rs.pth'
model_epoch = 240
model_accuracy = 93.68
model_loss = None
model_device = par.TRAIN_ARCH

used_model = mod.UsedModel(model_type,arg_load=True, arg_load_raw=True, arg_load_path=model_path, arg_load_device=model_device)
used_model.optimizer = optim.Adam(used_model.model.parameters(), lr=par.INITIAl_LEARNING_RATE)

# Create Savable copy of model
saved_model_states = copy.deepcopy(used_model.model.state_dict())
saved_optim_states = copy.deepcopy(used_model.optimizer.state_dict())

# Create Save Dictionary:
model_save_dict = \
    {
        'title': "This is save file of the model of the VisonNet - the PLN banknote recognition network",
        'name': model_save_path,
        'epoch': model_epoch,
        'accuracy': model_accuracy,
        'loss': model_loss,
        'optimizer': saved_optim_states,
        'model': saved_model_states
    }

# Save Current Model
torch.save(model_save_dict, model_save_path)

print("Model Resaved successfully")

