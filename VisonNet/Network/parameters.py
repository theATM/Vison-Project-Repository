import torch
import Network.Architecture.modeltype as modtype
from enum import Enum
"""
    Welcome to the params.py 
    Here you could and should change all the parameters available in the VisonNet
    First you can create your own Parameters Profile, this will be useful if you want to have multiple sets of parameters.
    Next make sure that all the paths are properly set up (especially the data files - go see the Data/index.txt for more)
    Than you should configure devices, batch sizes and amount of workers to make program work well on your machine.
    Have fun!
"""


# This is Enum class containing all parameter profiles.
class ParametersProfileType(Enum):
    POKISIEK = 1
    ATM = 2
    ATM_MOBILENET = 3
    # You can add new profile here
    # PROFILE_NAME = next free number


# Change to your profile
PARAMETERS_PROFILE = ParametersProfileType.ATM_MOBILENET
#PARAMETERS_PROFILE = ParametersProfileType.POKISIEK

# Check if profile is properly set up
if not isinstance(PARAMETERS_PROFILE, ParametersProfileType):
    print("Before launching anything, set your \"PARAMETERS_PROFILE\" first. To do it, go to the \"parameters.py\".")
    exit(-1)


########################################################################################################################

if PARAMETERS_PROFILE == ParametersProfileType.POKISIEK:

    #Data Parameters
    __data_path_prefix = "../.."
    __data_dir_name = "Data"
    __data_dataset_name = 'dataset'
    __data_testset_name = 'testset'
    __data_valset_name = 'valset'
    __data_dataset_batch_size = 16
    __data_valset_batch_size = 4
    __data_testset_batch_size = 4
    __data_dataset_num_workers = 8
    __data_valset_num_workers = 2
    __data_testset_num_workers = 2
    __data_single_batch_test_enable: bool = False

    ####################################################

    # Model Parameters
    __model_dir_name = '../../Models/'
    __model_file_type = '.pth'
    __model_additional_dirs = ""
    __model_load_name = 'Original_Resnet18_11-10-2021_21-21_Epoch_0020_Acc_21.95.pth'
    __model_load_raw_model_enable :bool = False
    __model_used_model_type = modtype.ModelType.Original_Mobilenet2

    ####################################################

    #Training Parameters
    __train_max_epoch_number = 400
    __train_device = 'cuda:0'
    __train_load_model_enable :bool = False
    __train_cudnn_enable = True
    __train_cudnn_benchmark_enable = True
    __train_initial_learning_rate = 0.01
    __train_scheduler_gamma = 0.8
    __train_eval_per_epochs = 20
    __train_grad_per_batch = 4

    ####################################################

    #Quantisation Parameters
    __quant_model_indir_name = "Quantin"
    __quant_model_outdir_name = "Quantout"
    __quant_model_name = 'Original_Resnet18_13-10-2021_07-44_Epoch_0380_Acc_89.71.pthEpoch_0240_Acc_93.68_rs.pth'
    __quant_save_model_name = 'ResQuant84First.pt'
    __quant_device = 'cpu'
    __quant_eval_enable = False
    __quant_dataset_batch_size = 4
    __quant_valset_batch_size = 4
    __quant_testset_batch_size = 4
    __quant_dataset_num_workers = 0
    __quant_valset_num_workers = 0
    __quant_testset_num_workers = 0

    ####################################################
    #Eval
    __eval_load_model_is_quantized = False


########################################################################################################################

elif PARAMETERS_PROFILE == ParametersProfileType.ATM:

    # Data Parameters
    __data_path_prefix = "../.."
    __data_dir_name = "Data"
    __data_dataset_name = 'dataset'
    __data_testset_name = 'testset'
    __data_valset_name = 'valset'


    __data_dataset_batch_size = 16
    __data_valset_batch_size = 4
    __data_testset_batch_size = 4

    __data_dataset_num_workers = 8
    __data_valset_num_workers = 2
    __data_testset_num_workers = 2


    __data_single_batch_test_enable :bool = False # Will run the model with only one batch to see if it works properly

    ####################################################

    # Model Parameters

    __model_dir_name = 'Models'
    __model_file_type = '.pth' # Model Presumed extension
    __model_additional_dirs = ""
    __model_load_name = 'Original_Mobilenet2_23-10-2021_16-38_Epoch_0004_Acc_20.07.pth'
    __model_load_raw_model_enable :bool = False
    __model_used_model_type = modtype.ModelType.Original_Mobilenet2


    ####################################################

    # Training Parameters
    __train_max_epoch_number = 800
    __train_device = 'cuda:0'  # for cpu type 'cpu', for gpu type 'cuda' to check for gpu enter to command prompt nvidia-smi
    __train_load_model_enable :bool = False
    __train_cudnn_enable = True
    __train_cudnn_benchmark_enable = True  # zysk +2% cuda  (?)
    __train_initial_learning_rate = 0.01
    __train_scheduler_gamma = 0.8
    __train_eval_per_epochs = 20
    __train_grad_per_batch = 4


    ####################################################

    # Quantisation Parameters
    __quant_model_indir_name = "Quantin"
    __quant_model_outdir_name = "Quantout"
    __quant_model_name = 'Original_Resnet18_13-10-2021_07-44_Epoch_0380_Acc_89.71.pthEpoch_0240_Acc_93.68_rs.pth'
    __quant_save_model_name =  'ResQuant84First.pt'
    __quant_device = 'cpu'
    __quant_eval_enable = False
    __quant_dataset_batch_size = 4
    __quant_valset_batch_size = 4
    __quant_testset_batch_size = 4
    __quant_dataset_num_workers = 0
    __quant_valset_num_workers = 0
    __quant_testset_num_workers = 0
    #Eval
    __eval_load_model_is_quantized = False


########################################################################################################################

elif PARAMETERS_PROFILE == ParametersProfileType.ATM_MOBILENET:
    # Data Parameters

    __data_path_prefix = "../.."
    __data_dir_name = "Data"
    __data_dataset_name = 'dataset'
    __data_testset_name = 'testset'
    __data_valset_name = 'valset'

    __data_dataset_batch_size = 16
    __data_valset_batch_size = 4
    __data_testset_batch_size = 4

    __data_dataset_num_workers = 8
    __data_valset_num_workers = 2
    __data_testset_num_workers = 2

    __data_single_batch_test_enable: bool = False  # Will run the model with only one batch to see if it works properly

    ####################################################

    # Model Parameters
    __model_path_prefix = "../.."
    __model_dir_name = 'Models'
    __model_additional_dirs = "/ModelType.Original_Mobilenet2_26-11-2021_08-33/"
    __model_load_name = 'ModelType.Original_Mobilenet2_26-11-2021_08-33Epoch_0240_Acc_95.60.pth'
    __model_file_type = '.pth'  # Model Presumed extension
    __model_load_raw_model_enable: bool = False
    __model_used_model_type = modtype.ModelType.Original_Mobilenet2

    ####################################################

    # Training Parameters
    __train_max_epoch_number = 800
    __train_device = 'cuda:0'  # for cpu type 'cpu', for gpu type 'cuda' to check for gpu enter to command prompt nvidia-smi
    __train_load_model_enable: bool = False
    __train_cudnn_enable = True
    __train_cudnn_benchmark_enable = True  # zysk +2% cuda  (?)
    __train_initial_learning_rate = 0.01
    __train_scheduler_gamma = 0.8
    __train_eval_per_epochs = 20
    __train_grad_per_batch = 4

    ####################################################

    # Quantisation Parameters
    __quant_model_indir_name = "Quantin"
    __quant_model_outdir_name = "Quantout"
    __quant_model_name = ""
    __quant_save_model_name = 'ResQuant84First.pt'
    __quant_device = 'cpu'
    __quant_eval_enable = False
    __quant_dataset_batch_size = 4
    __quant_valset_batch_size = 4
    __quant_testset_batch_size = 4
    __quant_dataset_num_workers = 0
    __quant_valset_num_workers = 0
    __quant_testset_num_workers = 0

    #Eval
    __eval_load_model_is_quantized = False

else: #None Init
    __data_path_prefix , __data_dir_name, __data_dataset_name, __data_testset_name, __data_valset_name, __data_dataset_batch_size, __data_valset_batch_size \
    , __data_testset_batch_size, __data_dataset_num_workers, __data_valset_num_workers, __data_testset_num_workers, __data_single_batch_test_enable \
    , __model_dir_name, __model_file_type,__model_additional_dirs, __model_load_name, __model_load_raw_model_enable, __model_used_model_type, __train_max_epoch_number \
    , __train_device, __train_load_model_enable, __train_cudnn_enable, __train_cudnn_benchmark_enable, __train_initial_learning_rate, __train_scheduler_gamma \
    , __train_eval_per_epochs, __train_grad_per_batch, __quant_model_indir_name, __quant_model_outdir_name, __quant_model_name, __quant_save_model_name \
    , __quant_device, __quant_eval_enable, __quant_dataset_batch_size, __quant_valset_batch_size, __quant_testset_batch_size, __quant_dataset_num_workers \
    , __quant_valset_num_workers, __quant_testset_num_workers, __eval_load_model_is_quantized = None


########################################################################################################################
############################################ Global Parameters #########################################################
########################################################################################################################

# Training Parameters
TRAIN_MAX_EPOCH_NUMBER : int = __train_max_epoch_number
TRAIN_DEVICE : str = __train_device  # for cpu type 'cpu', for gpu type 'cuda' to check for gpu enter to command prompt nvidia-smi
TRAIN_LOAD_MODEL_ENABLE : bool = __train_load_model_enable
torch.backends.cudnn.enabled = __train_cudnn_enable
torch.backends.cudnn.benchmark = __train_cudnn_benchmark_enable  # zysk +2% cuda  (?)
TRAIN_INITIAl_LEARNING_RATE : float = __train_initial_learning_rate
TRAIN_SCHEDULER_GAMMA : float = __train_scheduler_gamma
TRAIN_EVAL_PER_EPOCHS : int = __train_eval_per_epochs
TRAIN_GRAD_PER_BATCH : int = __train_grad_per_batch

####################################################

# Data Parameters
DATA_PATH_PREFIX : str = __data_path_prefix
DATA_DIR_NAME : str = __data_dir_name
DATA_DATASET_PATH : str = __data_path_prefix + '/' + __data_dir_name + '/' + __data_dataset_name
DATA_VALSET_PATH : str = __data_path_prefix + '/' + __data_dir_name + '/' + __data_valset_name
DATA_TESTSET_PATH : str = __data_path_prefix + '/' + __data_dir_name + '/' + __data_testset_name

DATA_DATASET_BATCH_SIZE : int = __data_dataset_batch_size
DATA_VALSET_BATCH_SIZE : int = __data_valset_batch_size
DATA_TESTSET_BATCH_SIZE : int = __data_testset_batch_size

DATA_DATASET_NUM_WORKERS : int = __quant_dataset_num_workers
DATA_VALSET_NUM_WORKERS : int = __data_valset_num_workers
DATA_TESTSET_NUM_WORKERS : int = __data_testset_num_workers

DATA_LOAD_RAW_MODEL_ENABLE : bool = __model_load_raw_model_enable
DATA_SINGLE_BATCH_TEST_ENABLE : bool = __data_single_batch_test_enable  # Will run the model with only one batch to see if it works properly

####################################################

# Model Parameters

MODEL_DIR_MODEL : str = __model_dir_name
MODEL_FILE_TYPE : str = __model_used_model_type  # Model Presumed extension
MODEL_LOAD_MODEL_PATH :str = __model_path_prefix + '/' + __model_dir_name + '/' + __model_additional_dirs +  __model_load_name
MODEL_USED_MODEL_TYPE : modtype.ModelType = __model_used_model_type


####################################################

# Quantisation Parameters
QUANT_MODEL_INDIR : str = __quant_model_indir_name
QUANT_MODEL_OUTDIR : str = __quant_model_outdir_name

QUANT_MODEL_PATH : str = __model_path_prefix + '/' + __model_dir_name + '/' + __quant_model_indir_name + '/' + __quant_model_name
QUANT_SAVE_MODEL_PATH : str = __model_path_prefix + '/' + __model_dir_name + '/' + __quant_model_outdir_name + '/' + __quant_model_name


QUANT_DEVICE : str = __quant_device
QUANT_EVAL_ENABLE : bool = __quant_eval_enable

QUANT_DATASET_BATCH_SIZE : int = __quant_dataset_batch_size
QUANT_VALSET_BATCH_SIZE : int = __quant_valset_batch_size
QUANT_TESTSET_BATCH_SIZE : int = __quant_testset_batch_size

QUANT_DATASET_NUM_WORKERS : int = __quant_dataset_num_workers
QUANT_VALSET_NUM_WORKERS : int = __quant_valset_num_workers
QUANT_TESTSET_NUM_WORKERS : int = __quant_testset_num_workers


####################################################
# Evaluation Parameters
EVAL_LOAD_MODEL_IS_QUANTIZED : bool = __eval_load_model_is_quantized