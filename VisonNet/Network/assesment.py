#import tkinter as tk
#from tkinter import ttk
#from PIL import Image,  ImageTk

import torch
import torchvision

from torchvision import transforms
from skimage import io
from PIL import Image#, ImageTk
#Prove of concept
import bankset as bank
import parameters as par
import model as mod

asses_loader = None
asses_data_path = ''

MODEL_PATH = '../Models/resnetTa94pretrained.pth'
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
            pred_val_batch, pred_batch = output_batch.topk(1, 1, True, True)
            for j, data in enumerate(input_batch):
                print(end='\n')
                input = input_batch.numpy()[j]
                label = label_batch.numpy()[j]
                name = name_batch[j]
                pred = pred_batch.numpy()[j][0]
                pred_val = pred_val_batch.numpy()[j][0]
                output = output_batch.numpy()[j]

                print("Next Image: " + name)
                print("Model Predictions:")
                print(["%.4f" % o for o in output])
                print("Available Classes:")
                print(["" + str(c) + "(" + str(k) + ")" + " " for c, k in bank.classes.items()])
                print("Choosen class nr " + str(pred) + " With sertanty = " + str(pred_val))

                pred_label = bank.anticlasses.get(pred.item())
                correct_val = output[label]

                print("Therefore image " + str(name) + " recognised as " + str(pred_label) + " zl, with " + str(
                    pred_val) + " certainty")

                if pred == label:
                    print("This is Correct")
                else:
                    print("This is Not Correct")

                print("This image should be " + "recognised as " + str(bank.anticlasses.get(label)) + " zl, "
                      + "class nr " + str(label) + " ( " + str(correct_val) + " )")




                user_choice = assesMenu()
                if user_choice == 0: continue
                elif user_choice == 1: return
                elif user_choice == 2:
                    #show image
                    img = Image.open(name)
                    img.show(title="Image " + name)
                    littleMenu()

            correct = pred.eq(label.contiguous().view(1, -1).expand_as(pred))
            correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
            print(str(correct_k))


    img = Image.open(my_image_path)
    img.show()




def assesMenu():
    print("To exit press q")
    print("To see picture press s")
    print("To continue press anything else")
    user_imput = input()
    if user_imput == 'q': return 1
    elif user_imput == 's': return 2
    return 0


def littleMenu():
    print("To continue press something")
    user_imput = input()


def assesMyImage():
    image = io.imread(my_image_path)
    image = transform_asses(image)
    image = image[None, :, :, :]

def assesNextImage():
    image = next(iter(asses_loader))


if __name__ == '__main__':
    main()