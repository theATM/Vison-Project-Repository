import torch
import random
from torch.utils.data import Dataset

import Network.Bank.bankset as bank
import Network.Bank.banksetuttils as bsu


class Erorset(Dataset): #Not Working
    def __init__(self, root_dirs, transform=None):
        self.images = []
        self.labels = []
        #Allow single dir string
        if type(root_dirs) == str:
            root_dirs = {root_dirs}
        #Load Images
        for root_dir in root_dirs:
            for value in bank.classes:
                tmpImages = bsu.listFilesInDir(f"{root_dir}/{value}/")
                self.images.extend(tmpImages)
                self.labels.extend([value for _ in range(len(tmpImages))])

        zipList = list(zip(self.images,self.labels))
        random.shuffle(zipList)
        self.images, self.labels = zip(*zipList)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imagePath = self.images[idx]
        label = self.labels[idx]
        image = torch.load(imagePath)

        if self.transform:
            image = self.transform(image)

        return image.to('cuda:0', non_blocking=True), label.to('cuda:0', non_blocking=True)