import Network.Architecture.modeltype as modtype

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
__data_single_batch_test_enable: bool = False

####################################################

# Model Parameters
__model_dir_name = '../../Models/'
__model_file_type = '.pth'
__model_additional_dirs = ""
__model_load_name = 'Original_Resnet18_11-10-2021_21-21_Epoch_0020_Acc_21.95.pth'
__model_load_raw_model_enable: bool = False
__model_used_model_type = modtype.ModelType.Original_Mobilenet2

####################################################

# Training Parameters
__train_max_epoch_number = 400
__train_device = 'cuda:0'
__train_load_model_enable: bool = False
__train_cudnn_enable = True
__train_cudnn_benchmark_enable = True
__train_initial_learning_rate = 0.01
__train_scheduler_gamma = 0.8
__train_eval_per_epochs = 20
__train_grad_per_batch = 4

####################################################

# Quantisation Parameters
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
# Eval
__eval_load_model_is_quantized = False
