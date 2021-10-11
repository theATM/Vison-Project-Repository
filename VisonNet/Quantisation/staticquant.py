import torch
import torchvision.models.quantization as quantmodels
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader

import Network.parameters as par
import Network.bankset as bank
import Network.model as mod



def quantMain():
    # Choose quantization engine
    if 'qnnpack' not in torch.backends.quantized.supported_engines:
        print("Unsuported backeng engine error - are you trying this on windows?")
        exit(-3)
    BACKEND_ENGINE = 'qnnpack'

    # Choose quantization device (cpu/gpu)
    quantDevice = torch.device('cpu')

    # Load Data
    dataset_loader, _, _ = bank.loadData(arg_load_train=True, arg_load_val=False, arg_load_test=False)






if __name__ == '__main__':
    quantMain()