# -*- coding: utf-8 -*-
# Author: yongyuan.name
# python index.py -database database -index featureCNN.h5
# python query_online.py -query ant.jpg -index featureCNN.h5 -result database
import os, _thread
import time

import h5py
import numpy as np
import argparse

from retrieval.extract_cnn_vgg16_keras import VGGNet
from tools.general import relpath_from_webpath, PathDict, is_allowed_ext
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 读取图片特征库
h5_file = h5py.File('retrieval/feature.h5', 'r')
feats = h5_file['feats'][:].tolist()
names = h5_file['names'][:]
names = [e.decode() for e in names]
h5_file.close()


'''
 Returns a list of filenames for all jpg images in a directory. 
'''


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if (is_allowed_ext(f))]


'''
 Extract features and index the images
'''


def purify_feature():
    for i, webpath in enumerate(names):
        relpath = relpath_from_webpath(webpath)
        if (not (relpath and os.path.exists(relpath) and webpath.split('/')[0] in PathDict)):
            print('remove ' + webpath)
            names.remove(webpath)
            feats.remove(feats[i])
    # 存入文件
    h5f = h5py.File('retrieval/feature.h5', 'w')
    h5f.create_dataset('feats', data=np.array(feats))
    h5f.create_dataset('names', data=names)
    h5f.close()


try:
    _thread.start_new_thread(purify_feature,())
except:
    print("净化图片特征线程启动失败")

def index_dir(webdir):
    relpath = PathDict[webdir]

    model = VGGNet()
    img_list = get_imlist(relpath)
    for i, img_path in enumerate(img_list):
        img_name = webdir + '/' + os.path.split(img_path)[1]
        if (img_name in names): continue
        norm_feat = model.extract_feat(img_path)
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d ,"+img_name+"; " + str(i + 1))

    h5f = h5py.File('retrieval/feature.h5', 'w')  # 存入文件
    h5f.create_dataset('feats', data=np.array(feats))
    h5f.create_dataset('names', data=names)
    h5f.close()

#检查新增图片并提取特证
t1 = time.process_time()
for webdir in PathDict:
    index_dir(webdir)
t2 = time.process_time()
print("index new images, done spent time: "+str(t2-t1))

if __name__ == "__main__":

    img_list = get_imlist('')

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))

    feats = np.array(feats)
    # print(feats)
    # directory for storing extracted features

    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File('feature.h5', 'w')
    h5f.create_dataset('dataset_1', data=feats)
    # h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()
