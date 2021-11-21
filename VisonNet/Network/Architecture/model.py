import os
import copy
import torch
import torch.optim as optim
import torchvision.models.quantization as originalModels

from torch import nn
from datetime import datetime
from torch.optim import lr_scheduler

#My Files
import Network.parameters as par
import Network.Architecture.mobilenet as mobilenet
from Network.Architecture.modeltype import ModelType


MODEL_ERROR_ID = -4


class UsedModel:

    def __init__(self, arg_choose_model: ModelType, arg_pretrained=False, arg_load_quantized=False,
                 arg_load=False, arg_load_raw=False, arg_load_path='', arg_load_device='', arg_remove_last_save=False):

        #Check if arguments make sense
        if arg_load_quantized is True and arg_load is False: exit(-1) # Cant quantize without loading a model
        if arg_load_raw is True and arg_load is False: exit(-1)

        #Public Variables, some will be initialized later
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0 #For Clean Start

        #Get current time to differentiate models
        current_time = datetime.now().strftime("%d.%m.%Y_%H:%M")

        #Create Future Save Path
        if arg_load is False:
            self.model_path, self.dir_path = self.__generateSaveTemplatePath(par.PATH_PREFIX, par.MODEL_DIR, arg_choose_model)
        else:
            self.model_path, self.dir_path = self.__generateSaveTemplatePathForLoad(arg_load_path)

        #Private Variables
        self.__reference_time = current_time #used to differentiate saved models from one training session from another
        self.__model_type = arg_choose_model #.name for string
        self.__model_file_type = par.MODEL_FILE_TYPE
        self.__model_dir = par.MODEL_DIR
        self.__model_saved = False
        self.__model_save_remove_last = arg_remove_last_save
        self.__model_save_epoch = None
        self.__model_save_acc = None
        self.__model_save_loss = None

        if arg_choose_model == ModelType.Original_Resnet18:
            print("Chosen Resnet18")
            # Prepare Model from Resnet
            original_model = originalModels.resnet18(pretrained=arg_pretrained, progress=True, quantize=False)
            self.model = self.__createCombinedModel(original_model)
            # Create Criterion and Optimizer
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=par.INITIAl_LEARNING_RATE)
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[32, 128, 160, 256, 512, 720], gamma=par.SCHEDULER_GAMMA)

        #elif chooseModel == ModelType.My_Resnet18:
        elif arg_choose_model == ModelType.Original_Mobilenet2:
            # Prepare Model from MobileNet
            self.model = mobilenet.MobileNetV2(num_classes=7)
            # Create Criterion and Optimizer
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[25, 60, 70, 80], gamma=0.1)

        else:
            print("Chosen Model Not Supported")
            exit(MODEL_ERROR_ID)

        #Load Model
        if arg_load is True:
            if arg_load_quantized is True: # Loading quantized model
                self.__loadQuantizedModel(arg_load_path)
            else: # Loading normal (float) model
                self.__loadModel(arg_load_path, arg_load_device, arg_load_raw=arg_load_raw)
                self.start_epoch = self.__model_save_epoch
        else:
            print("Model Created")


    @staticmethod
    def __createCombinedModel(arg_model_fe):
        model_fe_features = nn.Sequential(
            arg_model_fe.conv1,
            arg_model_fe.bn1,
            arg_model_fe.relu,
            arg_model_fe.maxpool,
            arg_model_fe.layer1,
            arg_model_fe.layer2,
            arg_model_fe.layer3,
            arg_model_fe.avgpool,
        )

        new_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(256, 7),
        )

        new_model = nn.Sequential(
            model_fe_features,
            nn.Flatten(1),
            new_head,
        )
        return new_model


    @staticmethod
    def __generateSaveTemplatePath(arg_path_prefix, arg_model_dir, arg_model_name: ModelType):
        """ It creates standard name for future saved model file
            This naming convention is used to differentiate between
            different trainings sessions and models used. Atm"""
        now = datetime.now()  # Used to differentiate saved models
        now_str = now.strftime("%d-%m-%Y_%H-%M")
        #Dir for all newly saved models
        save_dir_path= arg_path_prefix + '/' + arg_model_dir + '/' + str(arg_model_name) + '_' + now_str
        #Saced model template path
        save_template_path = save_dir_path + '/' + str(arg_model_name) + '_' + now_str
        return save_template_path, save_dir_path

    @staticmethod
    def __generateSaveTemplatePathForLoad(arg_load_path):
        """ It creates standard name for future saved model file
            created from loaded model. Atm"""
        now = datetime.now()  # Used to differentiate saved models
        now_str = now.strftime("%d.%m.%Y_%H:%M")
        # Break up load path into path and file name
        [dir_path, load_file_name] = arg_load_path.rsplit('/',1)
        save_dir_name = load_file_name.rsplit('_', 4)[0] + '_' + 'R' + '_' + now_str
        # Generate paths
        save_dir_path = dir_path + '/' + save_dir_name
        save_template_path = save_dir_path + '/' + save_dir_name
        return save_template_path, save_dir_path


    def __loadModel(self, arg_load_path, arg_load_device, arg_partial_load=True, arg_load_raw=False):
        if arg_load_path == '':
            print('Loading Model Path is Empty')
            exit(MODEL_ERROR_ID)
        if os.path.isfile(arg_load_path) is False:
            print('Loading Model File does not Exist')
            exit(MODEL_ERROR_ID)
        if arg_load_device == '':
            print('loading Model Device not Specified')
            exit(MODEL_ERROR_ID)

        #Load File:
        model_load_dict = torch.load(arg_load_path, map_location=arg_load_device)

        if arg_load_raw is False:
            #Deserialize:
            saved_model_states = model_load_dict['model']
            saved_optim_states = model_load_dict['optimizer']
            self.__model_save_epoch = model_load_dict['epoch']
            self.__model_save_acc = model_load_dict['accuracy']
            self.__model_save_loss = model_load_dict['loss']
        else:
            # For Bare Model Saves
            saved_model_states = model_load_dict
            saved_optim_states = None
            self.__model_save_epoch = None
            self.__model_save_acc = None
            self.__model_save_loss = None


        # Load Model:
        self.model.load_state_dict(saved_model_states, strict=not arg_partial_load)
        if saved_optim_states != None:
            self.optimizer.load_state_dict(saved_optim_states)
        print("Model Loaded")

    def __loadQuantizedModel(self, arg_load_path):
        if arg_load_path == '':
            print('Loading Model Path is Empty')
            exit(MODEL_ERROR_ID)
        if os.path.isfile(arg_load_path) is False:
            print('Loading Model File does not Exist')
            exit(MODEL_ERROR_ID)

        self.model = torch.jit.load(arg_load_path)
        print("Model Loaded")

    def saveModel(self, arg_epoch=None, arg_acc=None, arg_loss=None):
        """ Saves Model to File while training, can remove last save too (from the same model) """

        #Create Savable copy of model
        saved_model_states = copy.deepcopy(self.model.state_dict())
        saved_optim_states = copy.deepcopy(self.optimizer.state_dict())

        #Creates directory for all saved models
        if self.__model_saved is False:
            os.mkdir(self.dir_path)
            print("Created folder for the saved models")


        #Checks whether should remove last model first
        if self.__model_saved is True \
            and self.__model_save_remove_last is True \
            and self.__model_save_epoch >= 0 \
            and self.__model_save_acc >= 0:

            #Remove Last Model
            model_last_path = self.getSavedModelPath()
            #Check if Last Model File exist
            if os.path.isfile(model_last_path) is True:
                os.remove(model_last_path)
                print("Last Model Removed")
            else:
                print("Last Model Save File is Missing, cannot Remove it")


        # Update Training State Info
        self.__model_save_epoch = arg_epoch
        self.__model_save_acc = arg_acc
        self.__model_save_loss = arg_loss

        # Create Save Dictionary:
        model_save_dict = \
            {
                'title': "This is save file of the model of the VisonNet - the PLN banknote recognition network",
                'name': self.model_path,
                'epoch': self.__model_save_epoch,
                'accuracy': self.__model_save_acc,
                'loss': self.__model_save_loss,
                'optimizer': saved_optim_states,
                'model': saved_model_states
            }

        # Save Current Model
        model_save_path = self.getSavedModelPath()
        torch.save(model_save_dict, model_save_path)
        self.__model_saved = True  # set as saved
        print("Saved Current Model on Epoch " + str(self.__model_save_epoch + 1))

    def saveQuantizedModel(self, arg_save_model_path, arg_needed_dataset_loader):
        print("Started Saving Model")
        for i, data in enumerate(arg_needed_dataset_loader):
            inputs, labels = data['image'], data['class']
            traced_script_module = torch.jit.trace(self.model, inputs)
            traced_script_module.save(arg_save_model_path)
            break

        print('Model Saved Successfully')




    def getSavedModelPath(self):
        """ It constructs full model save path, by adding current training epoch and model accuracy to it. Atm"""
        return str(self.model_path) \
               + 'Epoch_' + str('%04d' % (self.__model_save_epoch+1)) \
               + '_Acc_' + str('%.2f' % self.__model_save_acc) \
               + str(self.__model_file_type)


    def printSizeofModel(self):
        torch.save(self.model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def fuzeModel(self):
        if self.__model_type == ModelType.Original_Resnet18:
            self.model = torch.quantization.fuse_modules(self.model, self.getLayersToFuse())
        elif self.__model_type == ModelType.Original_Mobilenet2:
            self.model.fuzeModel()
        else:
            print("Unrecognised model type")
            exit(-1)

    def getLayersToFuse(self):
        if self.__model_type == ModelType.Original_Resnet18:
            modules_to_fuse = [['1', '2'],
                               ['5.0.conv1', '5.0.bn1'],
                               ['5.0.conv2', '5.0.bn2'],
                               ['5.1.conv1', '5.1.bn1'],
                               ['5.1.conv2', '5.1.bn2'],

                               ['6.0.conv1', '6.0.bn1'],
                               ['6.0.conv2', '6.0.bn2'],
                               ['6.0.downsample.0', '6.0.downsample.1'],
                               ['6.1.conv1', '6.1.bn1'],
                               ['6.1.conv2', '6.1.bn2'],

                               ['7.0.conv1', '7.0.bn1'],
                               ['7.0.conv2', '7.0.bn2'],
                               ['7.0.downsample.0', '7.0.downsample.1'],
                               ['7.1.conv1', '7.1.bn1'],
                               ['7.1.conv2', '7.1.bn2']]
            return modules_to_fuse
        else:
            print("Unrecognised model type")
            exit(-1)


    def addQuantStubs(self):
        if self.__model_type == ModelType.Original_Resnet18:
            self.model
            new_model = nn.Sequential(
                torch.quantization.QuantStub(),
                self.model[0][0],
                self.model[0][1],
                self.model[0][2],
                self.model[0][3],
                nn.Sequential(
                    self.model[0][4][0],
                    self.model[0][4][1],
                ),
                nn.Sequential(
                    self.model[0][5][0],
                    self.model[0][5][1],
                ),
                nn.Sequential(
                    self.model[0][6][0],
                    self.model[0][6][1],
                ),
                self.model[0][7],
                self.model[1],
                self.model[2],
                torch.quantization.DeQuantStub(),
            )
            self.model = new_model
        elif self.__model_type == ModelType.Original_Mobilenet2:
            return
        else:
            print("Unrecognised model type")
            exit(-1)









