import tkinter as tk
from tkinter import ttk
from PIL import Image,  ImageTk

import torch
import torchvision
import tkinter as tk
from torchvision import transforms
from skimage import io
from PIL import Image, ImageTk
#Prove of concept
import bankset as bank
import parameters as par
import model as mod

asses_loader = None
asses_data_path = ''

MODEL_PATH = '../Models/OrgResnet18_11-09-2021_17-24_Epoch_0060_Acc_17.62.pth'
my_image_path = '../../../Data/testset/500/63.jpg'

transform_asses = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703])
])

root = None
button = None
picture = None
label = None

def main():
    print("Welcome to the assesment program")
    print("You can check how the model is doing in action")

    _, _, asses_loader = bank.loadData(arg_load_train=False, arg_load_val=False, arg_load_test=True)
    asses_device = torch.device(par.TRAIN_ARCH)
    if par.TRAIN_ARCH == 'cuda:0': torch.cuda.empty_cache()
    used_model = mod.UsedModel('Original_Resnet18', loadPath=MODEL_PATH, load=True)
    used_model.model.eval()
    print("Assesment Started")

    with torch.no_grad():
        for i, batch in enumerate(asses_loader):
            input_batch, label_batch, name_batch = batch['image'], batch['class'], batch['name']
            output_batch = used_model.model(input_batch)
            _, pred_batch = output_batch.topk(1, 1, True, True)
            #pred_batch = pred_batch.t()
            for j, data in enumerate(batch):
                input = input_batch.numpy()[j]
                label =label_batch.numpy()[j]
                name = name_batch[j]
                pred = pred_batch.numpy()[j][0]
                output = output_batch.numpy()[j]

                user_choice = assesMenu()
                if user_choice == 0: continue
                elif user_choice == 1: return
                #elif user_choice == 2:
                    #show image

            correct = pred.eq(label.contiguous().view(1, -1).expand_as(pred))
            correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
            print(str(correct_k))


    img = Image.open(my_image_path)
    img.show()




def assesMenu():
    print("To exit press q")
    print("To see picture press s")
    print("To continue press anything else")
    user_choice = input()
    if user_choice == 'q': return 1
    elif user_choice == 's': return 2
    return 0

def assesMyImage():
    image = io.imread(my_image_path)
    image = transform_asses(image)
    image = image[None, :, :, :]

def assesNextImage():
    image = next(iter(asses_loader))


if __name__ == '__main__':
    main()