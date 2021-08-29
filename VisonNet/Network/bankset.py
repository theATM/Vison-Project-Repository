import random
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
from os import listdir
from os.path import isfile, join


def listFilesInDir(dir):
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    return files


#Defines different banknotes denominations - recognised by network
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

    def __init__(self, root_dirs, transform=None):
        self.images = []
        #Allow single dir string
        if type(root_dirs) == str:
            root_dirs = {root_dirs}
        #Load Images
        for root_dir in root_dirs:
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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class SingleBatchTestSet(Dataset):

    def __init__(self, images, labels, names):
        'Initialization'
        self.images = images
        self.labels = labels
        self.names = names

    def __len__(self):
        'Returns the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select Image
        image = self.images[index]
        label = self.labels[index]
        name = self.names[index]
        item = {'image': image, 'class': label, 'name': name}

        return item