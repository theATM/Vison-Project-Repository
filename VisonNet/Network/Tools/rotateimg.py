import os
import Network.Bank.banksethelpers as bsh
import torchvision.transforms as T
from PIL import Image as pimg

INDIR = "../../Factory/rotateFactory/indir"
OUTDIR = "../../Factory/rotateFactory/outdir"

def main():
    '''This is the Rotation Tool - but can be used to alter images in other way
    Just insert your in and oud dirs, copy images to indir (they could be inside sub folders)
    and set up transforms you wish to perform, and run :) '''

    print("Hi this is image rotation tool")
    print("Remember to select your in and out dirs")
    if not os.path.isdir(INDIR):
        print("Wrong indir - cannot find")
        exit(-1);
    img_list, dir_list = listFilesNDirsInDir(INDIR)
    #prepare out dirs
    out_dir_list = [str(idir).replace(INDIR.split("/")[-1],OUTDIR.split("/")[-1]) for idir in dir_list]
    out_img_list = [str(idir).replace(INDIR.split("/")[-1],OUTDIR.split("/")[-1]) for idir in img_list]
    #Create out dirs
    for odir in out_dir_list:
        if not os.path.isdir(odir):
            os.mkdir(str(odir)) # this works because list is created recursively top bottom
    #Transform images
    for (img,oimg) in zip(img_list,out_img_list):
        inimg = pimg.open(img)
        resimg = my_Transform(inimg)
        resimg.save(oimg)
    print("Done",end="")



#Lists all files in dir - recursive
def listFilesNDirsInDir(dir):
    list_files = []
    list_dirs = [dir]
    for f in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,f)):
            list_files.append(os.path.join(dir,f))
        elif os.path.isdir(os.path.join(dir,f)):
            sub_files, sub_dirs = listFilesNDirsInDir(os.path.join(dir,f))
            list_files.extend(sub_files)
            list_dirs.extend(sub_dirs)
        else:
            pass
    return list_files, list_dirs

#Here you can set up any transforms you wish:
my_Transform = T.Compose([
    bsh.RandomRotationTransform(angles=[-90, 90, 0, 180, -180]),
])




if __name__ == '__main__':
    main()