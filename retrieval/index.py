# -*- coding: utf-8 -*-

import os, _thread,time,h5py,ssl
import numpy as np

from tools.general import relpath_from_webpath, PathDict, is_allowed_ext, executor,get_img_paths
from tools.val import retrieval_file_path,database_file_path

from remotes.RPC import extract_feat

ssl._create_default_https_context = ssl._create_unverified_context

feat_path = retrieval_file_path
# 读取图片特征库
names,feats = [],[]
# if os.path.exists(feat_path):
#     if(os.path.getsize(feat_path)<100):
#         os.remove(feat_path)
#     else:
#         h5_file = h5py.File(retrieval_file_path, 'r')
#         feats = h5_file['feats'][:].tolist()
#         names = h5_file['names'][:]
#         names = [e.decode() for e in names]
#         h5_file.close()

'''
 Returns a list of filenames for all jpg images in a directory. 
'''


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if (is_allowed_ext(f))]


'''
 Extract features and index the images
'''


def purify_feature():
    names1,feats1 = [],[]
    for i, webpath in enumerate(names):
        relpath = relpath_from_webpath(webpath)
        if not relpath:
            print('(retrieval) remove ' + webpath)
        else:
            names1.append(webpath)
            feats1.append(feats[i])

    names[:] = names1
    feats[:] = feats1
    # 存入文件
    h5f = h5py.File(retrieval_file_path, 'w')
    h5f.create_dataset('feats', data=np.array(feats))
    h5f.create_dataset('names', data=names)
    h5f.close()


def index_dir(webdir):
    relpath = PathDict[webdir]
    if not os.path.isdir(relpath):
        print('(retrieval) ' + webdir + ' 文件夹不存在')

    img_list = get_img_paths(relpath)
    for i, img_path in enumerate(img_list):
        img_name = webdir + '/' + os.path.split(img_path)[1]
        if (img_name in names): continue
        norm_feat = extract_feat(img_path)
        feats.append(norm_feat)
        names.append(img_name)
        print("(retrieval) extracting feature from image No. %d ," + img_name + "; " + str(i + 1))

    h5f = h5py.File(retrieval_file_path, 'w')  # 存入文件
    h5f.create_dataset('feats', data=np.array(feats))
    h5f.create_dataset('names', data=names)
    h5f.close()


def run():
    purify_feature()
    # 检查新增图片并提取特证
    t1 = time.process_time()
    for webdir in PathDict:
        index_dir(webdir)
    t2 = time.process_time()
    print("(retrieval) index new images, done spent time: " + str(t2 - t1))


executor.submit(run)

if __name__ == "__main__":

    img_list = get_imlist('')

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    names = []

    for i, img_path in enumerate(img_list):
        norm_feat = extract_feat(img_path)
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
