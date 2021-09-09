#TODO nie dizała jeszcze trzeba dostosować wszystkie pathy do naszych
from PIL import Image
import torch
from torchvision import transforms
import torchvision
from torch import nn


WEIGHTS_PATH = './best_model_rn.pth'
input_image = Image.open("testset/10/22.jpg")


device = torch.device("cpu")
preprocess = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(400),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

model.load_state_dict(torch.load(WEIGHTS_PATH))
model.to(device)
model.eval()

# traced_script_module = torch.jit.trace(model, input_batch)
# traced_script_module.save("rn18cpu.pt")

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')


with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))

