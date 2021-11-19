import torch
import random
from torch.utils.data import Dataset

import Network.Bank.bankset as bank
import Network.Bank.banksetuttils as bsu


class SingleBatchTestSet(Dataset):


    def __init__(self, arg_seed_trainloader):
        #Load
        single_batch = next(iter(arg_seed_trainloader))
        single_batch_images, single_batch_labels, single_batch_names = \
            single_batch['image'], single_batch['class'], single_batch['name']

        #'Initialization'
        self.images = single_batch_images
        self.labels = single_batch_labels
        self.names = single_batch_names

    def __len__(self):
        #'Returns the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select Image
        image = self.images[index]
        label = self.labels[index]
        name = self.names[index]
        item = {'image': image, 'class': label, 'name': name}

        return item

