import os

import Network.Bank.bankset as bank

# Utilities Functions used in Bankset:

# returns Dir name for eg. "/home/user/funny" -> "funny"
def pickDirNameFromDirPath(dir):
    return dir.split("/")[-1]


#Lists all files in dir - recursive
def listFilesInDir(dir):
    list = []
    for f in os.listdir(dir):
        if os.path.isfile(os.path.join(dir,f)):
            list.append(os.path.join(dir,f))
        elif os.path.isdir(os.path.join(dir,f)):
            list.extend(listFilesInDir(os.path.join(dir,f)))
        else:
            pass
    return list


#Returns the list of dirs (dir paths)
def listDirsInDir(dir):
    return [str(dir + "/" + d) for d in os.listdir(dir) if not os.path.isdir(d)]


#Checks if dir contains "class dirs" for eg. directory with name "10", "20" or "none:
def checkDirIfContainsAllClasses(dir):
    return True if len([subdir for subdir in listDirsInDir(dir) if pickDirNameFromDirPath(subdir) in bank.classes]) == 7 else False


#Returns list of Subdirs if they are not classes and when ty=hey are it returns the list with the arg file in it
def listSubdirsIfPresent(dir):
    return listDirsInDir(dir) if not checkDirIfContainsAllClasses(dir) else [dir]


def listFilesInDirPol(dir, label):
    labels = []
    files = []
    for f in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, f)):
            files.append(os.path.join(dir, f))
            labels.append(label)
    return files, labels



