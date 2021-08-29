import torch

#Training Parameters
MAX_EPOCH_NUMBER = 105
TRAIN_ARCH = 'cuda:0'  # for cpu type 'cpu', for gpu type 'cuda'
LOAD_MODEL = False
torch.backends.cudnn.enabled = True;
torch.backends.cudnn.benchmark = True; #zysk +2% cuda  (?) ale wywala sie (?)
INITIAl_LEARNING_RATE = 0.01
EVAL_PER_EPOCHS = 20

#Data Parameters
BEST_MODEL_PATH = '../Models/best_model.pth'
DATASET_PATH = '../Data/dataset'
TESTSET_PATH = '../Data/testset'
VALSET_PATH = '../Data/valset'

#Quantisation Parameters


