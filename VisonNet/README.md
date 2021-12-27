## VisonNet - Polish Banknote Recognition Neural Network
This is directory with implementation of a neural network training program.
In this repository you can find script to train ResNet18 and Mobilenetv2
based neural network to recognize polish banknotes. 
Also we've attached quantisation script which helps in neural net static quantisation (Resnet18 only)
to reduce it size and give performance boost on mobile devices.

### Requirements
This program uses Python 3.7.12
Preferred OS is Linux (for windows there is a dockerfile)
Used with CUDA 11.4 and CuDNN 8.2.4
Pytorch 1.9.1+cu111, torchvision 0.10.1+cu111, torchaudio 0.9.1

To install Pytorch
linux:
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
windows:
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio===0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

Rest of requirements (not mentioned yet) are in requirements.txt to install it use
pip3 install requirements.txt
Best to use with venv environment

### Usable scripts:

All runnable scripts have main function attached 
Look for ^__main__^

#### Major ones are in Functional directory
- training.py
- evaluate.py
- staticquant.py

#### Minor ones are in Tools directory
- assesment-py
- imgtest.py
- resave.py
- rotateimg.py
- transplot.py

### Data:
- Images of PLN banknotes 244:244p
Datasets should be inserted into 
VisonNet/Data directory
(create dataset, testset and valset)

Link to data:
www.kaggle.com/dataset/99636797e98397fa6161113867b50bed663be07900885852cda713b1cc76e52d


### Parameters:
- To train efficiently create Parameter Profile or use default
to modify program behaviour

### To use scripts:
- run them in Pycharm or other IDE, or in python 

### Results:
- The team trained Resnet18 model to 96.5% accuracy after quantisation
- The team trained Mobilenetv2 model to 93% accuracy after quantisation

### Authors:
- Aleksander Madajczak
- Kamil Pokornicki
- Karol Dziki

### Continuation
If one wants to continue this project, advised is to get in touch with authors

### Credits:
This program is based on Polish Banknote Recognition Neural Network

Linked by https://www.kaggle.com/bartomiejgawrych/polish-banknotes-polskie-banknoty
