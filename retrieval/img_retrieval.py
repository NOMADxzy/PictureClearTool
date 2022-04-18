from flask import Blueprint,request

from retrieval.extract_cnn_vgg16_keras import VGGNet
from tools.general import relpath_from_webpath,thumbnail_from_webpath,get_tag,HOST,webpath_belongto_dir,\
    get_thumbnail_pic,get_img_detail,executor,settings,webpath_from_relpath
from tools.val import database_file_path
import numpy as np
from retrieval.index import names,feats,index_dir
import h5py
import os,sqlite3
import json
from pathlib import Path
#检索图片相关的api在这里
retrievalapp = Blueprint('img_retrieval',__name__)


model = VGGNet()


@retrievalapp.route('/',methods=['POST','GET'])
def retrieval():
    if(len(feats)==0): return {'num':0,'candicates':[]}
    max_res = 8
    min_res = 2
    threshold = settings['rela']
    print('img retrieval thres = ' + str(threshold))
    query = request.json['query']#目标图片的相对路径
    print('query img '+query)
    qvec = model.extract_feat(query)
    scores = np.dot(qvec,np.asarray(feats).T)
    ranked_idx = np.argsort(scores)[::-1]

    candicates,num = [],0
    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()
    for i in range(max_res):
        id = ranked_idx[i]
        webpath = names[id]
        score = scores[id]
        if(score<threshold and i>=min_res): break
        relpath = relpath_from_webpath(webpath)
        im = {'id': relpath,
              'index': num,
              # 'thumbnail': 'atom:///' + relpath,
              # 'original': 'atom:///' + relpath,
              'thumbnail': HOST + thumbnail_from_webpath(webpath),
              'original': HOST + webpath,
              'score':score,
              'details': get_img_detail(relpath,cursor),
              'tags': get_tag(webpath)}
        if(im['id']==query): continue
        candicates.append(im)#去掉同一图片
        num += 1
    # if(relpath_from_webpath(candicates[0][0])==query): candicates = candicates[1:]
    detect.close()
    return {'num':num,'candicates':candicates}
@retrievalapp.route('/cpt_all/<path:dir>',methods=['GET','POST'])
def cpt_all(dir):
    thre = settings['rela']
    names0 = names
    if request.method == 'POST':
        relpaths = request.json['paths']
        webpaths = [webpath_from_relpath(relpath) for relpath in relpaths]

        feats1,names1 = [],[]
        for i,name in enumerate(names0):
            if(name in webpaths):
                feats1.append(feats[i])
                names1.append(name)

    else:
        feats1,names1 = feats,names

    if(len(feats1)==0):
        return {'total': 0, 'relative': []}
    filt = True
    dir_len = len(dir)
    if dir=='__all__': filt = False
    f = np.asarray(feats1)
    mat = np.dot(f,f.T)
    mat = np.triu(mat,0)#取上三角(包含对角线)，过滤重复检测
    args = np.argwhere(mat >= thre)
    rela_imgs,cur = [[]],0
    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()
    for arg in args:
        if(cur<arg[0]):#按行分组
            cur += 1
            rela_imgs.append([])
        if(os.path.exists(relpath_from_webpath(names1[arg[1]]))):#该图片存在
            if filt:#指定文件夹的相似图片
                if names1[arg[1]][0:dir_len]==dir: rela_imgs[cur].append((names1[arg[1]],mat[arg[0],arg[1]]))
                #不需要指定文件夹的相似图片
            else: rela_imgs[cur].append((names1[arg[1]],mat[arg[0],arg[1]]))


    #删除图片数不足两张的
    purifed_imgs,num = [],0
    for img_list in rela_imgs:
        length = len(img_list)
        if(length>1):
            img_list = [{
                'id':relpath_from_webpath(img_score[0]),
                'index':num+i,
                'thumbnail': HOST + thumbnail_from_webpath(img_score[0]),
                'original':HOST + img_score[0],
                # 'thumbnail': 'atom:///' + relpath_from_webpath(img_score[0]),
                # 'original': 'atom:///' + relpath_from_webpath(img_score[0]),
                'tags':get_tag(img_score[0]),
                'checked':not i==0,
                'details': get_img_detail(relpath_from_webpath(img_score[0]),cursor),
                'score':img_score[1]
            } for i,img_score in enumerate(img_list)]
            purifed_imgs.append(img_list)
            num+=length
    detect.close()
    return {'total':num,'relative':purifed_imgs}
