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

# Check if profile is properly set up
if not isinstance(PARAMETERS_PROFILE, ParametersProfileType):
    print("Before launching anything, set your \"PARAMETERS_PROFILE\" first. To do it, go to the \"parameters.py\".")
    exit(-1)


########################################################################################################################

if PARAMETERS_PROFILE == ParametersProfileType.POKISIEK:

    import Network.Parameters.pokisiek as user

########################################################################################################################

elif PARAMETERS_PROFILE == ParametersProfileType.ATM:

    import Network.Parameters.atmresnet as user

########################################################################################################################

elif PARAMETERS_PROFILE == ParametersProfileType.ATM_MOBILENET:

    import Network.Parameters.atmmobile as user

else: #None Init

    import Network.Parameters.none as user


########################################################################################################################
############################################ Global Parameters #########################################################
########################################################################################################################

# Training Parameters
TRAIN_MAX_EPOCH_NUMBER : int =              user.__train_max_epoch_number
TRAIN_DEVICE : str =                        user.__train_device  # for cpu type 'cpu', for gpu type 'cuda' to check for gpu enter to command prompt nvidia-smi
TRAIN_LOAD_MODEL_ENABLE : bool =            user.__train_load_model_enable
torch.backends.cudnn.enabled =              user.__train_cudnn_enable
torch.backends.cudnn.benchmark =            user.__train_cudnn_benchmark_enable  # zysk +2% cuda  (?)
TRAIN_INITIAl_LEARNING_RATE : float =       user.__train_initial_learning_rate
TRAIN_SCHEDULER_GAMMA : float =             user.__train_scheduler_gamma
TRAIN_EVAL_PER_EPOCHS : int =               user.__train_eval_per_epochs
TRAIN_GRAD_PER_BATCH : int =                user.__train_grad_per_batch
TRAIN_PRINT_PER_BATCH :int =                user.__train_print_per_batch
TRAIN_MILESTONES =                          user.__train_milestones

####################################################

# Data Parameters
DATA_PATH_PREFIX : str =                    user.__data_path_prefix
DATA_DIR_NAME : str =                       user.__data_dir_name
DATA_DATASET_PATH : str =                   user.__data_path_prefix + '/' \
                                          + user.__data_dir_name + '/' \
                                          + user.__data_dataset_name

DATA_VALSET_PATH : str =                    user.__data_path_prefix + '/' \
                                          + user.__data_dir_name + '/' \
                                          + user.__data_valset_name

DATA_TESTSET_PATH : str =                   user.__data_path_prefix + '/' \
                                          + user.__data_dir_name + '/' \
                                          + user.__data_testset_name

DATA_DATASET_BATCH_SIZE : int =             user.__data_dataset_batch_size
DATA_VALSET_BATCH_SIZE : int =              user.__data_valset_batch_size
DATA_TESTSET_BATCH_SIZE : int =             user.__data_testset_batch_size

DATA_DATASET_NUM_WORKERS : int =            user.__data_dataset_num_workers
DATA_VALSET_NUM_WORKERS : int =             user.__data_valset_num_workers
DATA_TESTSET_NUM_WORKERS : int =            user.__data_testset_num_workers

DATA_LOAD_RAW_MODEL_ENABLE : bool =         user.__model_load_raw_model_enable
DATA_SINGLE_BATCH_TEST_ENABLE : bool =      user.__data_single_batch_test_enable  # Will run the model with only one batch to see if it works properly

####################################################

# Model Parameters

MODEL_DIR_MODEL : str =                     user.__model_dir_name
MODEL_FILE_TYPE : str =                     user.__model_file_type  # Model Presumed extension

MODEL_LOAD_MODEL_PATH :str =                user.__model_path_prefix + '/' \
                                          + user.__model_dir_name + '/' \
                                          +(user.__model_additional_dirs+"/"
                                         if user.__model_additional_dirs != "" else "") \
                                          + user.__model_load_name

MODEL_USED_MODEL_TYPE : modtype.ModelType = user.__model_used_model_type


####################################################

# Quantisation Parameters
QUANT_MODEL_INDIR : str =                   user.__quant_model_indir_name
QUANT_MODEL_OUTDIR : str =                  user.__quant_model_outdir_name

QUANT_MODEL_PATH : str =                    user.__model_path_prefix + '/' \
                                          + user.__model_dir_name + '/' \
                                          + user.__quant_model_indir_name + '/' \
                                          + user.__quant_model_name

QUANT_SAVE_MODEL_PATH : str =               user.__model_path_prefix + '/' \
                                          + user.__model_dir_name + '/' \
                                          + user.__quant_model_outdir_name + '/' \
                                          + user.__quant_save_model_name


QUANT_DEVICE : str =                        user.__quant_device
QUANT_EVAL_ENABLE : bool =                  user.__quant_eval_enable

QUANT_DATASET_BATCH_SIZE : int =            user.__quant_dataset_batch_size
QUANT_VALSET_BATCH_SIZE : int =             user.__quant_valset_batch_size
QUANT_TESTSET_BATCH_SIZE : int =            user.__quant_testset_batch_size

QUANT_DATASET_NUM_WORKERS : int =           user.__quant_dataset_num_workers
QUANT_VALSET_NUM_WORKERS : int =            user.__quant_valset_num_workers
QUANT_TESTSET_NUM_WORKERS : int =           user.__quant_testset_num_workers


####################################################
# Evaluation Parameters
EVAL_LOAD_MODEL_IS_QUANTIZED : bool =       user.__eval_load_model_is_quantized
LOAD_SUPER_RAW = False