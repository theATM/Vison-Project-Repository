# import tkinter as tk
# from tkinter import ttk
# from PIL import Image,  ImageTk

import torch
import torchvision

from torchvision import transforms
from skimage import io
from PIL import Image  # , ImageTk
# Prove of concept
import Network.Bank.bankset as bank
import Network.Bank.transforms as trans
import Network.parameters as par
import Network.Architecture.model as mod
import Network.Architecture.modeltype as modtype

MODEL_PATH = "/home/olek/Projects/VisonProject/VisonNet/Models/Original_Resnet18_07-12-2021_23-36/SixthDotSevHalGood190ResQuant97.pt"
    #"/home/olek/Projects/VisonTestLoadAp/app/src/main/assets/FourthGood240ResQuant92.pt"
#'../../Models/Original_Resnet18_02-12-2021_20-43/Quant_Original_Resnet18_02-12-2021_20-43Last_Epoch_0241_Acc_81.95.pt'
MY_IMAGE_PATH = "/home/olek/Projects/VisonTestLoadAp/app/src/main/assets/20211204_190935.jpg"
#"/home/olek/Desktop/LasTest/20211204_190935.jpg"
 #"/home/olek/Pictures/VisonData/TrainingData/TrainDataStorage/100/atmImg100/atmImg100_Normal/20210923_220318c.jpg"
#"/home/olek/Desktop/LasTest/20211204_190935.jpg"
#"/home/olek/Desktop/LasTest/20211204_154215.jpg"
#'../Data/testset/500/63.jpg'
SHOW_ONLY_NEGATIVE = True #used when assesing many images  - the program would stop only on those predicted wrong
SCAN_FOR_WORST = True #used to show the worst pedicted picture in dataset
PRINT_BAD = False
MODEL_TYPE = modtype.ModelType.Original_Resnet18
transform_asses = trans.TRANSFORM_BLANK
LOAD_DEVICE = par.TRAIN_DEVICE
LOAD_QUANTIZED = True

def main():
    print("Welcome to the assessment program")
    print("You can check how the model is doing in action")

    asses_device = torch.device(par.TRAIN_DEVICE)
    if par.TRAIN_DEVICE == 'cuda:0': torch.cuda.empty_cache()
    # Load Model
    used_model = mod.UsedModel(MODEL_TYPE, arg_load_path=MODEL_PATH, arg_load=True,arg_load_device=LOAD_DEVICE,arg_load_quantized=LOAD_QUANTIZED)

    # Choose what to do
    user_choice = mainMenu()

    if user_choice == 1:
        print("Chosen Single Image Assessment")
        print("Assessment Started")
        assesOneImage(used_model, MY_IMAGE_PATH)
        return

    elif user_choice == 0:
        print("Chosen Multi Image Assessment")
        _, _, asses_loader  = bank.loadData(arg_load_train=False, arg_load_val=False, arg_load_test=True)
        print("Assessment Started")
        assesManyImages(used_model, asses_loader)
        return

    elif user_choice == -1:
        print("Bye")
        return

    else:
        print("Bye?")
        return


def assesOneImage(used_model, image_path):

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703])
        ])

    image = io.imread(image_path)
    image = transform_test(image)
    image = image[None,:,:,:]

    image_label = "5" #TODO

    used_model.model.eval()
    with torch.no_grad():
        output = used_model.model(image)
        pred_val, pred = output.topk(1, 1, True, True)
        #pred_val = pred_val.numpy()
        printAssesResults(image_path, int(image_label), output.numpy()[0], pred.numpy()[0][0], pred_val.numpy()[0][0])

    img = Image.open(image_path)
    img.show(title="Image " + image_path)


def assesManyImages(used_model, asses_loader):
    worst_image = None
    all_count = 0
    all_by_class = [0,0,0,0,0,0,0]
    bad_count = 0
    bad_by_class = [0,0,0,0,0,0,0]
    bad_by_file = dict()
    used_model.model.eval()
    with torch.no_grad():
        if PRINT_BAD: print("Bad images will be printed")
        for i, batch in enumerate(asses_loader):
            input_batch, label_batch, name_batch = batch['image'], batch['class'], batch['name']
            output_batch = used_model.model(input_batch)
            pred_val_batch, pred_batch = output_batch.topk(1, 1, True, True)
            for j, data in enumerate(input_batch):
                if SCAN_FOR_WORST is False: print(end='\n')
                input = input_batch.numpy()[j]
                label = label_batch.numpy()[j]
                name = name_batch[j]
                pred = pred_batch.numpy()[j][0]
                pred_val = pred_val_batch.numpy()[j][0]
                output = output_batch.numpy()[j]

                all_count+=1
                all_by_class[label]+=1

                if SHOW_ONLY_NEGATIVE and pred == label:
                    continue

                if pred != label:
                    bad_count+=1
                    bad_by_class[label]+=1
                    dir_name = name.split("/")[-2]
                    if dir_name in bad_by_file: bad_by_file[dir_name] += 1
                    else: bad_by_file[dir_name] = 1

                if PRINT_BAD and pred != label :
                    printAssesResults(name, label, output, pred, pred_val)

                if SCAN_FOR_WORST:
                    if pred != label and (worst_image == None or pred_val < worst_image[4] ):
                        worst_image = (name, label, output, pred, pred_val )
                    continue

                if PRINT_BAD is False or pred == label:
                    printAssesResults(name, label, output, pred, pred_val)

                user_choice = assesMenu()
                if user_choice == 0:
                    continue
                elif user_choice == -1:
                    return
                elif user_choice == 1:
                    # show image
                    img = Image.open(name)
                    img.show(title="Image " + name)
                    continueMenu()

        if SCAN_FOR_WORST:
            print("Worst Image in Set")
            printAssesResults(worst_image[0], worst_image[1], worst_image[2], worst_image[3], worst_image[4])
            img = Image.open(worst_image[0])
            img.show(title="Image " + worst_image[0])

    print(end="\n")
    print(f"Checked {len(asses_loader)} images")
    print(f"Founded {bad_count} not recognised correctly")
    print(f"That leaves {len(asses_loader)-bad_count} images recognised correctly")
    print(f"Because there are {all_count} in this dataset")
    print(f"Bad images by class")
    print(bad_by_class)
    print(f"All images by class")
    print(all_by_class)
    print("There are all dirs with bad images")
    print(bad_by_file)
    print(end="\n")


def printAssesResults(image_name, image_label, output, pred, pred_val):
    print("\nImage: " + image_name)
    print("Model Predictions:")
    print(["%.4f" % o for o in output])
    print("Available Classes:")
    print(["" + str(c) + "(" + str(k) + ")" + " " for c, k in bank.classes.items()])
    print("Chosen class nr " + str(pred) + " With certainty = " + str(pred_val))

    pred_label = bank.anticlasses.get(pred.item())
    correct_val = output[image_label]

    print("Therefore image " + str(image_name) + " recognised as " + str(pred_label) + " zl, with " + str(
        pred_val) + " certainty")

    if pred == image_label:
        print("This is Correct")
    else:
        print("This is Not Correct")

    print("This image should be " + "recognised as " + str(bank.anticlasses.get(image_label)) + " zl, "
          + "class nr " + str(image_label) + " ( " + str(correct_val) + " )")
    print(end="\n")

#Menu's:

def mainMenu():
    print("Choose what you want to do")
    print("To exit press q")
    print("Press 1 to test model on specific picture")
    print("Press anything else to test model on specified dataset")
    user_imput = input()
    if user_imput == 'q':
        return -1
    elif user_imput == '1':
        return 1
    return 0


def assesMenu():
    print("To exit press q")
    print("To see picture press s")
    print("To continue press anything else")
    user_imput = input()
    if user_imput == 'q':
        return -1
    elif user_imput == 's':
        return 1
    return 0


def continueMenu():
    print("To continue press something")
    user_input = input()


if __name__ == '__main__':
    main()
