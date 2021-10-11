import os
import copy
import torch
import torch.optim as optim
import torchvision.models.quantization as originalModels

from torch import nn
from datetime import datetime

#My Files
import Network.parameters as par
import Network.OldCode.mobilenet_v2 as oldModelsMobileNetv2

MODEL_ERROR_ID = -4


class UsedModel:

    def __init__(self, chooseModel, pretrained=False, quantize=False, load=False, loadPath='', remove_last_save=True):

        #Public Variables, some will be initialized later
        self.model = None
        self.criterion = None
        self.optimizer = None
        if load is False:
            self.model_path =  self.__generateInitPath(par.MODEL_DIR, chooseModel)
        else:
            self.model_path = loadPath
        #Private Variables
        self.__model_name = chooseModel
        self.__model_file_type = par.MODEL_FILE_TYPE
        self.__model_dir = par.MODEL_DIR
        self.__model_saved = False
        self.__model_save_remove_last = remove_last_save
        self.__model_save_epoch = -1
        self.__model_save_acc = -1.0

        if chooseModel == 'Original_Resnet18':
            print("Chosen Resnet18")
            # Prepare Model from Resnet
            original_model = originalModels.resnet18(pretrained=pretrained, progress=True, quantize=quantize)
            self.model = self.__createCombinedModel(original_model)
            if load is True: # to retrain / finetune
                self.__loadModel(loadPath)
            else:
                print("Model Created")
            # Create Criterion and Optimizer
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=par.INITIAl_LEARNING_RATE)

        #elif chooseModel == 'My_Resnet18':
        elif chooseModel == 'Old_Mobilenet2':
            # Prepare Model from MobileNet
            self.model = oldModelsMobileNetv2.MobileNetV2(num_classes=7)
            # Create Criterion and Optimizer
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        else:
            print("Chosen Model Not Supported")
            exit(MODEL_ERROR_ID)


    @staticmethod
    def __createCombinedModel(model_fe):
        model_fe_features = nn.Sequential(
            model_fe.conv1,
            model_fe.bn1,
            model_fe.relu,
            model_fe.maxpool,
            model_fe.layer1,
            model_fe.layer2,
            model_fe.layer3,
            model_fe.avgpool,
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
    def __generateInitPath(model_dir, model_name):
        """ It creates standard name for saved model file
            (Directory) / Current Time _ Model Name _
            Rest of the path will be added later and will consist of
            Training Epoch _ Model Accuracy . File Type
            This naming convention is used to differentiate between
            different trainings sessions and models used. Atm"""
        now = datetime.now()  # Used to differentiate saved models
        now_str = now.strftime("%d-%m-%Y_%H-%M")
        name_str = model_dir + str(model_name) + '_' + now_str + '_'
        return name_str


    def __loadModel(self, loadPath):
        if loadPath == '':
            print('Loading Model Path is Empty')
            exit(MODEL_ERROR_ID)
        if os.path.isfile(loadPath) is False:
            print('Loading Model File does not Exist')
            exit(MODEL_ERROR_ID)
        self.model.load_state_dict(torch.load(loadPath,map_location=torch.device('cpu')), strict=False)
        print("Model Loaded")


    def saveModel(self, arg_epoch, arg_acc):
        """ Saves Model to File while training, can remove last save too (from the same model) """

        #Create Savable copy of model
        savedModel = copy.deepcopy(self.model.state_dict())

        #Checks whether should remove last model first
        if  self.__model_saved is True \
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

        #Update Training State Info
        self.__model_save_epoch = arg_epoch
        self.__model_save_acc = arg_acc

        #Save Current Model
        model_save_path = self.getSavedModelPath()
        torch.save(savedModel, model_save_path)
        print("Saved Current Model on Epoch " + str(self.__model_save_epoch+1))


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





