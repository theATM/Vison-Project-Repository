import torch
from PIL import Image
import torchvision
from torchvision import transforms
from torch import nn
import bankset_old as oldBank
from torch.utils.data import DataLoader
from mobilenet_v2 import MobileNetV2, AverageMeter


WEIGHTS_PATH = oldBank.BEST_MODEL_DIR + 'best_test.pth'
def test():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform_test = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(400),
                                        transforms.CenterCrop((400,400)),
                                        transforms.ToTensor(),])
                                        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    testset = oldBank.Bankset(oldBank.TESTSET_PATH, transform_test)
    testloader = DataLoader(testset, batch_size=6, shuffle=True, num_workers=4)

    model = MobileNetV2(num_classes = 7)

    #num_ftrs = model.classifier[1].in_features
    #model.classifier[1] = nn.Linear(num_ftrs, 7)
    model.load_state_dict(torch.load('best_quantized.pth'))
    model.to(device)
    model.eval()

    running_corrects = 0
    matrix = torch.zeros([7,7])
    for i, data in enumerate(testloader, 0):
            inputs, labels = data['image'], data['class']
            inputs = inputs.to(device)
            labels = labels.to(device)


            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                for i, p in enumerate(preds):
                    matrix[labels[i]][p] += 1
            running_corrects += torch.sum(preds == labels.data)
    print(matrix)
    acc = running_corrects.double() / len(testset)
    print('Test Acc: {:.4f}'.format(acc))

test()
