import random
from torch.utils.data import Dataset
import torch
from skimage import io, transform
from os import listdir
from os.path import isfile, join

def listFilesInDir(dir):
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    return files



#Defines diferent banknotes denominations - rocognised by network
classes = {
    '10':  0,
    '20':  1,
    '50':  2,
    '100': 3,
    '200': 4,
    '500': 5,
    'none': 6,
}

#Class containing one set of images.
class Bankset(Dataset):
    #Usefull Paths
    BEST_MODEL_PATH = '../Models/best_model.pth'
    DATASET_PATH = '../Data/dataset'
    TESTSET_PATH = '../Data/testset'
    VALSET_PATH = '../Data/valset'

    def __init__(self, root_dir, transform=None):
        self.images = []
        #Load Images
        for value in classes:
            tmp = map(lambda x: {'class': value, 'image':x}, listFilesInDir(f"{root_dir}/{value}/"))
            self.images.extend(tmp)

        random.shuffle(self.images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]['image']
        image = io.imread(img_name)

        item = {'image': image, 'class': classes[self.images[idx]['class']], 'name': img_name}
        if self.transform:
            item['image'] = self.transform(item['image'])

        return item
