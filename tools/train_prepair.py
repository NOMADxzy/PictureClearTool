import os,yaml
import shutil
import sqlite3
import pathlib
from tools.val import train_dic,cls_idx_base
from tools.general import info

def data2txt(d1, d2, d3, d4,txtpath):
    fd = open(txtpath, 'a')
    fd.write('0' + " " + str(d1) + " " + str(d2) + " " + str(d3) + " " + str(d4) + "\n")
    fd.close()
def writetxt(box,txtpath):
    clsName, xmin, ymin, xmax, ymax = box

    xcenter = (xmax + xmin) / 2
    ycenter = (ymax + ymin) / 2
    width = abs(xmax - xmin)
    height = abs(ymax - ymin)

    data2txt(xcenter, ycenter, width, height, txtpath)

def add_to_train(relpath,boxs):
    for box in boxs:
        clsName = box[0]
        if not os.path.exists(train_dic + '/' + clsName):
            os.mkdir(train_dic + '/' + clsName)
        if not os.path.exists(train_dic + '/' + clsName+'/images'):
            os.mkdir(train_dic + '/' + clsName+'/images')
        if not os.path.exists(train_dic + '/' + clsName+'/labels'):
            os.mkdir(train_dic + '/' + clsName+'/labels')

        with open(train_dic+'/'+clsName+'/'+'data.yaml', 'w', encoding='utf8') as f:
            info['names'] = [clsName]
            info['nc'] = 1
            info['path'] = './data/fromuser/'+clsName
            yaml.dump(info, f, allow_unicode=True)
            f.close()

        train_images_path = train_dic +'/'+clsName+ '/images'
        train_labels_path = train_dic +'/'+clsName+ '/labels'

        num = len(list(pathlib.Path(train_labels_path).glob('*.txt')))

        filetype = '.'+relpath.rsplit('.',1)[-1]
        imagepath = os.path.join(train_images_path, str(num).zfill(5)+filetype)
        txtpath = os.path.join(train_labels_path, str(num).zfill(5)+'.txt')

        writetxt(box,txtpath)
        shutil.copy(relpath, imagepath)


