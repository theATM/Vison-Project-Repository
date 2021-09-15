import tkinter as tk
from tkinter import ttk
from PIL import Image,  ImageTk







import torch
import torchvision
import tkinter as tk
from torchvision import transforms
from skimage import io
from PIL import Image, ImageTk
#Prove of concept

asses_loader = None
asses_data_path = ''

my_image_path = ''

transform_asses = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48269427, 0.43759444, 0.4045701], std=[0.24467267, 0.23742135, 0.24701703])
])

root = None
button = None
picture = None
label = None

def main():
    root = tk.Tk()
    button = tk.Button(root, text="Hejo", command=assesMyImage)
    #picture = tk.PhotoImage( Image.open('../../../Data/testset/200/49.jpg'))
    #picture = tk.PhotoImage(file='chocolate-bar.png')
    img = Image.open('49.jpg')
    #img.save('fafa.png')
    picture = ImageTk.PhotoImage(img)
    label = tk.Label(root,image=picture)
    label.image = picture
    label.pack()
    button.pack()
    root.mainloop()

def assesMyImage():
    image = io.imread(my_image_path)
    image = transform_asses(image)
    image = image[None, :, :, :]

def assesNextImage():
    image = next(iter(asses_loader))


if __name__ == '__main__':
    main()