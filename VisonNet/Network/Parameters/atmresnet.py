''' This is default and working parameter profile for Resnet18,
    best accuracy = 97% (unquantized) and 96.5% (quantized)'''



import Network.Architecture.modeltype as modtype

# Data Parameters
__data_path_prefix = "../.."
__data_dir_name = "Data"
__data_dataset_name = 'dataset'
__data_testset_name = 'testset'
__data_valset_name = 'valset'

__data_dataset_batch_size = 16
__data_valset_batch_size = 8
__data_testset_batch_size = 4

__data_dataset_num_workers = 8
__data_valset_num_workers = 4
__data_testset_num_workers = 2

__data_single_batch_test_enable: bool = False  # Will run the model with only one batch to see if it works properly - must set train_grad_per_batch to 1

####################################################

# Model Parameters
__model_path_prefix = "../.."
__model_dir_name = 'Models'
__model_file_type = '.pth'  # Model Presumed extension
__model_additional_dirs = "Original_Resnet18_07-12-2021_23-36"
__model_load_name = 'Original_Resnet18_07-12-2021_23-36_Epoch_0190_Acc_97.56.pth'
__model_load_raw_model_enable: bool = False
__model_used_model_type = modtype.ModelType.Original_Resnet18

####################################################

# Training Parameters
__train_max_epoch_number = 200  # 240 #800
__train_device = 'cuda:0'  # for cpu type 'cpu', for gpu type 'cuda' to check for gpu enter to command prompt nvidia-smi
__train_load_model_enable: bool = False
__train_cudnn_enable = True
__train_cudnn_benchmark_enable = True  # zysk +2% cuda  (?)
__train_initial_learning_rate = 0.001  # must be min 0.00(..)
__train_scheduler_gamma = 0.75  # 0.5 #0.2
__train_eval_per_epochs = 10
__train_grad_per_batch = 8
__train_print_per_batch = 256
__train_milestones = [50, 100, 150]  # [40,60,80] #[40, 80, 120, 180]#[60, 120, 200, 400]#[32, 128, 160, 256, 512, 720]

####################################################

# Quantisation Parameters
__quant_model_indir_name = "Quantin"
__quant_model_outdir_name = "Quantout"
__quant_model_name = 'Original_Resnet18_07-12-2021_23-36_Epoch_0190_Acc_97.56.pth'
__quant_save_model_name = 'Original_Resnet18_07-12-2021_23-36_Epoch_0190_Acc_97.56.pth'
__quant_device = 'cpu'
__quant_eval_enable = False
__quant_dataset_batch_size = 8
__quant_valset_batch_size = 8
__quant_testset_batch_size = 4
__quant_dataset_num_workers = 0
__quant_valset_num_workers = 0
__quant_testset_num_workers = 0
# Eval
__eval_load_model_is_quantized = False