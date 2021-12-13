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
__train_max_epoch_number = 200 #400
__train_device = 'cuda:0'  # for cpu type 'cpu', for gpu type 'cuda' to check for gpu enter to command prompt nvidia-smi
__train_load_model_enable: bool = False
__train_cudnn_enable = True
__train_cudnn_benchmark_enable = True  # zysk +2% cuda  (?)
__train_initial_learning_rate = 0.002 #0.015
__train_scheduler_gamma = 0.8
__train_eval_per_epochs = 1 #20
__train_grad_per_batch = 8#4
__train_print_per_batch = 256#128
__train_milestones = [80,120,160]#[80, 160, 240, 300]

####################################################

# Quantisation Parameters
__quant_model_indir_name = "Quantin"
__quant_model_outdir_name = "Quantout"
__quant_model_name = "Original_Mobilenet2_13-12-2021_16-41_Epoch_0004_Acc_79.08.pth"
__quant_save_model_name = 'Mobile50QuantMy92.pt'
__quant_device = 'cpu'
__quant_eval_enable = False
__quant_dataset_batch_size = 8
__quant_valset_batch_size = 4
__quant_testset_batch_size = 4
__quant_dataset_num_workers = 0
__quant_valset_num_workers = 0
__quant_testset_num_workers = 0

#Eval
__eval_load_model_is_quantized = False