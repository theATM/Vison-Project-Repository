#https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
import random

from PIL import Image, ImageStat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


import torch
import torchvision.transforms as T
import torchvision.transforms.functional as Tf
import Network.Bank.banksethelpers as bsh
import Network.Bank.transforms as trans

plt.rcParams["savefig.bbox"] = 'tight'
org_img = Image.open("/home/olek/Pictures/VisonData/TrainingData/Datasets/OriginalDataset/10/originalImg10/1.jpg")
my_img = Image.open("/home/olek/Pictures/VisonData/TrainingData/Datasets/IndependentDataset2/10/atmImg10_Normal/20211002_205902c.jpg")
dark_img = Image.open("/home/olek/Pictures/VisonData/TrainingData/Datasets/IndependentDataset2/10/20211104_222556_desent_10_dark_meh/20211104_222556s_064c.png")
dark_lamp_img = Image.open("/home/olek/Pictures/VisonData/TrainingData/Datasets/IndependentDataset2/10/20211104_222706_face_10_lamp_darkTrick/20211104_222706s_003c.png")
dark_lown_img = Image.open("/home/olek/Pictures/VisonData/TrainingData/TrainDataStorage/10/atmImg10/atmImg10_Dark/20210529_222951c.jpg")
czerwonyKwiat = Image.open("/home/olek/Pictures/Globe/czerwonyKwiat.jpeg")


# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def ogrPlot(orig_img,imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()



def plot(sour_img, imgs):

    num_rows = int ((len(imgs) + 1) / 5) + ((len(imgs) + 1) % 5)
    num_cols = 5
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)


    axs[0, 0].set(title='Original image')
    axs[0, 0].title.set_size(8)
    axs[0,0].imshow(sour_img)
    axs[0,0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    img_idx = 0
    for row_idx in range(num_rows):
        for col_idx in range (num_cols):
            if row_idx == 0 and col_idx == 0 : continue
            if img_idx >= len(imgs) :
                ax = axs[row_idx, col_idx]
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], frame_on=False)
                continue
            axs[row_idx, col_idx].imshow( imgs[img_idx] )
            axs[row_idx, col_idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[] )
            img_idx += 1

    plt.tight_layout()
    plt.show()


my_Transform = T.Compose([
    T.Resize(224),
    T.CenterCrop((224, 224)),
    bsh.RandomRotationTransform(angles=[-90, 90, 0, 180, -180]),
    #bsh.CustomColorJitter(contrast=(0.7,1.3),brightness=(0.4,1.45),saturation=(0.8,1.2)),
    bsh.CustomColorJitter(contrast=(0.7,1.3),brightness=(0.4,1.45),saturation=(0.8,1.2)),
    #T.transforms.RandomPerspective(distortion_scale=0.5, p=0.5 , fill=0),
    T.transforms.ToTensor(),
    #T.transforms.RandomApply(
    #        [bsh.AddGaussianNoise(0.5, 0.7)],
    #        p=0.25
    #    ),
    T.transforms.GaussianBlur((5,9),(1,5)),
    #T.transforms.RandomErasing(p=0.5, scale=(0.02, 0.08), ratio=(0.5, 2.3), value='random'),
    T.transforms.ToPILImage(),

])

TRANSFORM_BLANK = \
    T.transforms.Compose([
        #T.transforms.ToPILImage(),
        T.transforms.Resize(224),
        T.transforms.CenterCrop((224, 224)),
        #T.transforms.ToTensor(),
        #T.transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701],
        #                                 std=[0.24467267, 0.23742135, 0.24701703]),
        #T.transforms.ToPILImage(),
    ])

TRANSFORM_QUANTIZE = TRANSFORM_BLANK
TRANSFORM_EVAL = TRANSFORM_BLANK

TRANSFORM_TRAIN = \
    T.transforms.Compose([
        #T.transforms.ToPILImage(),
        T.transforms.Resize(224),
        T.transforms.CenterCrop((224, 224)),
        bsh.RandomRotationTransform(angles=[-90, 90, 0, 180, -180]), #Rotates randomly by 90 degrees - keeps whole image inside circle
        bsh.CustomColorJitter(contrast=(0.7,1.3),brightness=(0.4,1.45),saturation=(0.8,1.2)),
        # T.transforms.ToTensor(),
        #T.transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701],
        #                                std=[0.24467267, 0.23742135, 0.24701703]),

        #T.transforms.ToPILImage(),
    ])


#sub_imgs = [my_Transform(my_img) for _ in range(19)]
#plot(my_img,sub_imgs)
#im = my_img.convert('L')
#stat = ImageStat.Stat(im)
#print(stat.rms[0])
#
#sub_imgs = [my_Transform(org_img) for _ in range(19)]
#plot(org_img,sub_imgs)
#im = org_img.convert('L')
#stat = ImageStat.Stat(im)
#print(stat.rms[0])
#
#sub_imgs = [my_Transform(dark_img) for _ in range(19)]
#plot(dark_img,sub_imgs)
#im = dark_img.convert('L')
#stat = ImageStat.Stat(im)
#print(stat.rms[0])
#
#sub_imgs = [my_Transform(dark_lamp_img) for _ in range(19)]
#plot(dark_lamp_img,sub_imgs)
#im = dark_lamp_img.convert('L')
#stat = ImageStat.Stat(im)
#print(stat.rms[0])
#
#sub_imgs = [my_Transform(dark_lown_img) for _ in range(19)]
#plot(dark_lown_img,sub_imgs)
#im = dark_lown_img.convert('L')
#stat = ImageStat.Stat(im)
#print(stat.rms[0])
#
sub_imgs = [TRANSFORM_BLANK(czerwonyKwiat) for _ in range(4)]
plot(czerwonyKwiat,sub_imgs)
im = czerwonyKwiat.convert('L')
stat = ImageStat.Stat(im)
print(stat.rms[0])

sub_imgs = [TRANSFORM_TRAIN(czerwonyKwiat) for _ in range(19)]
plot(czerwonyKwiat,sub_imgs)
im = czerwonyKwiat.convert('L')
stat = ImageStat.Stat(im)
print(stat.rms[0])