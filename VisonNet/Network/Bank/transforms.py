import torchvision
import Network.Bank.banksethelpers as bsh


TRANSFORM_BLANK = \
    torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701],
                                         std=[0.24467267, 0.23742135, 0.24701703])
    ])

TRANSFORM_QUANTIZE = TRANSFORM_BLANK
TRANSFORM_EVAL = TRANSFORM_BLANK

TRANSFORM_TRAIN = \
    torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop((224, 224)),
        bsh.RandomRotationTransform(angles=[-90, 90, 0, 180, -180]), #Rotates randomly by 90 degrees - keeps whole image inside circle
        bsh.CustomColorJitter(contrast=(0.7,1.3),brightness=(0.4,1.45),saturation=(0.8,1.2)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701],
                                         std=[0.24467267, 0.23742135, 0.24701703]),
    ])
