import os
import shutil
import sqlite3
import pathlib
from tools.general import names
from tools.val import train_dic,cls_idx_base


train_images_path = train_dic + '/images'
train_labels_path = train_dic + '/labels'

def data2txt(d1, d2, d3, d4, clsName,txtpath):
    fd = open(txtpath, 'a')
    index = clsName - cls_idx_base
    fd.write(str(index) + " " + str(d1) + " " + str(d2) + " " + str(d3) + " " + str(d4) + "\n")
    fd.close()
def writetxt(boxs,txtpath):
    for box in boxs:
        clsName,xmin, ymin, xmax, ymax = box
        if(clsName<cls_idx_base) : continue #只要新模型中的类

        xcenter = (xmax + xmin) / 2
        ycenter = (ymax + ymin) / 2
        width = abs(xmax - xmin)
        height = abs(ymax - ymin)

        data2txt(xcenter, ycenter, width, height, clsName, txtpath)

def add_to_train(relpath,boxs):
    num = len(list(pathlib.Path(train_labels_path).glob('*.txt')))

    filetype = '.'+relpath.rsplit('.',1)[-1]
    imagepath = os.path.join(train_images_path, str(num).zfill(5)+filetype)
    txtpath = os.path.join(train_labels_path, str(num).zfill(5)+'.txt')
    
    writetxt(boxs,txtpath)
    shutil.copy(relpath, imagepath)


