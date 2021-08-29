import os
import torch
import torchvision.models.quantization as originalModels
from torchvision.models import MobileNetV2
from torch import nn


def create(load = False, loadPath = ''):
    # crate model from resnet
    original_model = originalModels.resnet18(pretrained=True, progress=True, quantize=False)
    model = create_combined_model(original_model)
    if load == True:
        # to retrain / finetune
        if loadPath == '':
            print('Loading Path is Empty')
            exit(-4)
        model.load_state_dict(torch.load(loadPath), strict=False)
        print("Model Loaded")
    else:
        print("Model Created")
    return model


def create_combined_model(model_fe):
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


def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')