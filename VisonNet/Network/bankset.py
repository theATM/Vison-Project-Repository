import random
import torch
import torchvision
from torch.utils.data import Dataset
from skimage import io, transform
from os import listdir
from os.path import isfile, join
import parameters as par
from torchvision import transforms
from torch.utils.data import DataLoader


def listFilesInDir(dir):
    files = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    return files


def listFilesInDirPol(dir, label):
    labels = []
    files = []
    for f in listdir(dir):
        if isfile(join(dir, f)):
            files.append(join(dir, f))
            labels.append(label)
    return files, labels


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

anticlasses = {
      0: 10,
      1: 20,
      2: 50,
      3: 100,
      4: 200,
      5: 500,
      6: 'none',
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



#Loading Data Function
def loadData(arg_load_train = True, arg_load_val = True,arg_load_test = True,
             arg_trans_train = None, arg_trans_val = None, arg_trans_test = None,
             single_batch_test = False):

    #Create Default Transformations:
    if arg_trans_train == None and arg_load_train is True:
        transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        RandomRotationTransform(angles=[-90, 90, 0, 180, -180]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.75, 1.45), contrast=0.5, saturation=0.5, hue=0.3),
        torchvision.transforms.RandomApply(
            [torchvision.transforms.Grayscale(num_output_channels=3)],
            p=0.35
        ),
        transforms.ToTensor(),
        torchvision.transforms.RandomApply(
            [AddGaussianNoise(0., 1.)],
            p=0.45
        ),
        transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.08), ratio=(0.5, 2.3), value='random'),
    ])

    if arg_trans_val == None and arg_load_val is True:
        transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703])
    ])

    if arg_trans_test == None and arg_load_test is True:
        transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703])
    ])

    # Load Data
    trainset = None
    valset = None
    testset = None
    trainloader = None
    valloader = None
    testloader = None
    print("Loaded",end=' ')
    if arg_load_train is True:
        trainset = Bankset(par.DATASET_PATH, transform_train)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
        print("TrainSet",end=' ')
    if arg_load_val is True:
        valset = Bankset(par.VALSET_PATH, transform_val)
        valloader = DataLoader(valset, batch_size=4, shuffle=True, num_workers=2)
        print("ValSet", end=' ')
    if arg_load_test is True:
        testset = Bankset(par.TESTSET_PATH, transform_test)
        testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
        print("TestSet", end=' ')
    if (arg_load_train or  arg_load_val or arg_load_test) is False:
        print("No Data")
        return
    else:
        print("Successfully")

    if single_batch_test == True:
        #Preform Single Batch Test
        singleBatch = next(iter(trainloader))
        singleBatchImages, singleBatchLabels, singleBatchNames = singleBatch['image'], singleBatch['class'], singleBatch['name']
        singleBatchDataSet = SingleBatchTestSet(singleBatchImages, singleBatchLabels, singleBatchNames)
        trainloader = DataLoader(singleBatchDataSet, batch_size=6, shuffle=True, pin_memory=True, num_workers=4)
        print("Single Batch Test Chosen")

    return  trainloader, valloader, testloader



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




class Erorset(Dataset): #Not Working
    def __init__(self, root_dirs, transform=None):
        self.images = []
        self.labels = []
        #Allow single dir string
        if type(root_dirs) == str:
            root_dirs = {root_dirs}
        #Load Images
        for root_dir in root_dirs:
            for value in classes:
                tmpImages = listFilesInDir(f"{root_dir}/{value}/")
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