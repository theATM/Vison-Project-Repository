#https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
import random

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as Tf

plt.rcParams["savefig.bbox"] = 'tight'
org_img = Image.open("/home/olek/Pictures/VisonData/dataset/OriginalDataset/10/1.jpg")
my_img = Image.open("/home/olek/Pictures/VisonMore/DataStorage/10/atmImg10_Normal/20211002_205902c.jpg")
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(sour_img, imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [sour_img] + row if with_orig else row
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
    plt.show()


my_Transform = T.Compose([
    T.Resize(224),
    T.CenterCrop((224, 224)),
    T.F
])

img = Tf.rotate(my_img,random.uniform(0,90))

rotater = T.RandomRotation(1800)
rotated_imgs = [my_Transform(img) for _ in range(1)]
plot(img,rotated_imgs)