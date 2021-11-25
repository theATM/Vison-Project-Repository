import torch
import random
import skimage
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage import io

import Network.parameters as par
import Network.Bank.transforms as trans
import Network.Bank.banksetuttils as bsu
import Network.Bank.banksethelpers as bsh
import Network.Bank.singlebatchset as sbt


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
        # Allow single dir string
        if type(root_dirs) == str:
            root_dirs = {root_dirs}
        # Load Images
        for root_dir in root_dirs:
            #Check for subdirs
            for sub_dir in bsu.listSubdirsIfPresent(root_dir):
                for value in classes:
                    tmp = map(lambda x: {'class': value, 'image': x}, bsu.listFilesInDir(f"{sub_dir}/{value}/"))
                    self.images.extend(tmp)

        random.shuffle(self.images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]['image']
        image = skimage.io.imread(img_name)

        item = {'image': image, 'class': classes[self.images[idx]['class']], 'name': img_name}
        if self.transform:
            item['image'] = self.transform(item['image'])

        return item


#Loading Data Function
def loadData(arg_load_train = True, arg_load_val = True, arg_load_test = True,
             arg_trans_train = None, arg_trans_val = None, arg_trans_test = None,
             single_batch_test = False, quantisation_mode = False):

    # Create Default Transformations:
    transform_train = arg_trans_train
    transform_val = arg_trans_val
    transform_test = arg_trans_test

    if arg_trans_train is None and arg_load_train is True:
        transform_train = trans.TRANSFORM_TRAIN

    if arg_trans_val is None and arg_load_val is True:
        transform_val = trans.TRANSFORM_BLANK

    if arg_trans_test is None and arg_load_test is True:
        transform_test = trans.TRANSFORM_BLANK

    # Load Data
    trainset = None
    valset = None
    testset = None
    trainloader = None
    valloader = None
    testloader = None
    print("Loaded",end=' ')


    if quantisation_mode is True:
        train_batch_size = par.QUANT_DATASET_BATCH_SIZE
        train_num_workers = par.QUANT_DATASET_NUM_WORKERS
        val_batch_size = par.QUANT_VALSET_BATCH_SIZE
        val_num_workers = par.QUANT_VALSET_NUM_WORKERS
        test_batch_size = par.QUANT_TESTSET_BATCH_SIZE
        test_num_workers = par.QUANT_TESTSET_NUM_WORKERS
    else:
        train_batch_size = par.DATASET_BATCH_SIZE
        train_num_workers = par.DATASET_NUM_WORKERS
        val_batch_size = par.VALSET_BATCH_SIZE
        val_num_workers = par.VALSET_NUM_WORKERS
        test_batch_size = par.TESTSET_BATCH_SIZE
        test_num_workers = par.TESTSET_NUM_WORKERS


    if arg_load_train is True:
        trainset = Bankset(par.DATASET_PATH, transform_train)
        trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=train_num_workers)
        print("TrainSet",end=' ')
    if arg_load_val is True:
        valset = Bankset(par.VALSET_PATH, transform_val)
        valloader = DataLoader(valset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers)
        print("ValSet", end=' ')
    if arg_load_test is True:
        testset = Bankset(par.TESTSET_PATH, transform_test)
        testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=test_num_workers)
        print("TestSet", end=' ')
    if (arg_load_train or  arg_load_val or arg_load_test) is False:
        print("No Data")
        return
    else:
        print("Successfully")

    if single_batch_test is True:
        #Preform Single Batch Test
        single_batch_dataset = sbt.SingleBatchTestSet(trainloader)
        trainloader = DataLoader(single_batch_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
        print("Single Batch Test Chosen")

    return trainloader, valloader, testloader



