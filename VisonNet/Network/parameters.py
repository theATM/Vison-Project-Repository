import torch
from enum import Enum
"""
    Welcome to the params.py 
    Here you could and should change all the parameters available in the VisonNet
    First you can create your own Parameters Profile, this will be useful if you want to have multiple sets of parameters.
    Next make sure that all the paths are properly set up (especially the data files - go see the Data/index.txt for more)
    Than you should configure devices, batch sizes and amount of workers to work well on your machine.
    Have fun!
"""


# This is Enum class containing all parameter profiles.
class ParametersProfileType(Enum):
    POKISIEK = 1
    ATM = 2
    # You can add new profile here
    # PROFILE_NAME = next free number


# Change to your profile
PARAMETERS_PROFILE = ParametersProfileType.ATM
#PARAMETERS_PROFILE = ParametersProfileType.POKISIEK

# Check if profile is properly set up
if not isinstance(PARAMETERS_PROFILE, ParametersProfileType):
    print("Before launching anything, set your \"PARAMETERS_PROFILE\" first. To do it, go to the \"parameters.py\".")
    exit(-1)


########################################################################################################################

if PARAMETERS_PROFILE == ParametersProfileType.POKISIEK:

    #Training Parameters
    MAX_EPOCH_NUMBER = 400
    TRAIN_ARCH = 'cuda:0'  # for cpu type 'cpu', for gpu type 'cuda' to check for gpu enter to command prompt nvidia-smi
    LOAD_MODEL = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True #zysk +2% cuda  (?)
    INITIAl_LEARNING_RATE = 0.01
    SCHEDULER_GAMMA = 0.8
    EVAL_PER_EPOCHS = 20
    GRAD_PER_BATCH = 4

    ####################################################

    #Data Parameters
    MODEL_DIR = '../Models/'
    MODEL_NAME = 'OrgResnet18'
    MODEL_FILE_TYPE = '.pth'
    BEST_MODEL_PATH = '../Models/Original_Resnet18_11-10-2021_21-21_Epoch_0020_Acc_21.95.pth' #Original_Resnet18_08-10-2021_02-05_Epoch_0001_Acc_17.83.pth'
    DATASET_PATH = '../Data/dataset'
    TESTSET_PATH = '../Data/testset'
    VALSET_PATH = '../Data/valset'

    DATASET_BATCH_SIZE = 16
    VALSET_BATCH_SIZE = 4
    TESTSET_BATCH_SIZE = 4

    DATASET_NUM_WORKERS = 8
    VALSET_NUM_WORKERS = 2
    TESTSET_NUM_WORKERS = 2




    ####################################################

    #Quantisation Parameters
    QUANT_MODEL_PATH = 'resnetTa94pretrained.pth' #'../Models/Original_Resnet18_11-10-2021_21-21_Epoch_0020_Acc_21.95.pth'
    QUANT_DEVICE = 'cpu'
    DO_EVALUATE = False

########################################################################################################################

if PARAMETERS_PROFILE == ParametersProfileType.ATM:
    # Training Parameters
    MAX_EPOCH_NUMBER = 400
    TRAIN_ARCH = 'cuda:0'  # for cpu type 'cpu', for gpu type 'cuda' to check for gpu enter to command prompt nvidia-smi
    LOAD_MODEL = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # zysk +2% cuda  (?)
    INITIAl_LEARNING_RATE = 0.01
    SCHEDULER_GAMMA = 0.8
    EVAL_PER_EPOCHS = 20
    GRAD_PER_BATCH = 4

    ####################################################

    # Data Parameters
    MODEL_DIR = '../Models/'
    MODEL_NAME = 'OrgResnet18'
    MODEL_FILE_TYPE = '.pth'
    BEST_MODEL_PATH = '../Models/Original_Resnet18_11-10-2021_21-21_Epoch_0020_Acc_21.95.pth'  # Original_Resnet18_08-10-2021_02-05_Epoch_0001_Acc_17.83.pth'
    DATASET_PATH = '../Data/dataset'
    TESTSET_PATH = '../Data/testset'
    VALSET_PATH = '../Data/valset'

    DATASET_BATCH_SIZE = 16
    VALSET_BATCH_SIZE = 4
    TESTSET_BATCH_SIZE = 4

    DATASET_NUM_WORKERS = 8
    VALSET_NUM_WORKERS = 2
    TESTSET_NUM_WORKERS = 2

    ####################################################

    # Quantisation Parameters
    QUANT_MODEL_PATH = 'resnetTa94pretrained.pth'  # '../Models/Original_Resnet18_11-10-2021_21-21_Epoch_0020_Acc_21.95.pth'
    QUANT_DEVICE = 'cpu'
    DO_EVALUATE = False
