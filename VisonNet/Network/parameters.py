import torch

#Training Parameters
MAX_EPOCH_NUMBER = 105
TRAIN_ARCH = 'cuda:0'  # for cpu type 'cpu', for gpu type 'cuda' to check for gpu enter to command prompt nvidia-smi
LOAD_MODEL = False
torch.backends.cudnn.enabled = True;
torch.backends.cudnn.benchmark = True; #zysk +2% cuda  (?)
INITIAl_LEARNING_RATE = 0.01
SCHEDULER_GAMMA = 0.8
EVAL_PER_EPOCHS = 20
GRAD_PER_BATCH = 4

#Data Parameters
MODEL_DIR = '../Models/'
MODEL_NAME = 'OrgResnet18'
MODEL_FILE_TYPE = '.pth'
BEST_MODEL_PATH = '../Models/Original_Resnet18_08-10-2021_02-05_Epoch_0001_Acc_17.83.pth'
DATASET_PATH = '../Data/dataset'
TESTSET_PATH = '../Data/testset'
VALSET_PATH = '../Data/valset'

#Quantisation Parameters


