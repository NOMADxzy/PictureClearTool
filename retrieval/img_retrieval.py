import os,sqlite3
from flask import Blueprint,request
import numpy as np
from PIL import Image

from tools.general import relpath_from_webpath,thumbnail_from_webpath,get_tag,HOST,webpath_belongto_dir,\
    get_thumbnail_pic,get_img_detail,executor,settings,webpath_from_relpath
from tools.val import database_file_path

# from retrieval.index import names,feats,index_dir,extract_feat
from remotes.RPC import extract_feat
from detects.blur import imageFeatures

# imageFeatures = ImageFeatures()
names,feats,cornerfeats = imageFeatures.names,imageFeatures.feats,imageFeatures.cornerfeats

#检索图片相关的api在这里
retrievalapp = Blueprint('img_retrieval',__name__)



@retrievalapp.route('/index/<path:dir>',methods=['GET'])
def index(dir):
    print('computing new dir vgg feature')
    # total = executor.submit(index_dir,dir)
    return 'ok',200

@retrievalapp.route('/',methods=['POST','GET'])
def retrieval():
    if(len(feats)==0):
        candicates = []
        return {'num':0,'candicates':candicates}
    max_res = len(feats)
    min_res = 0
    if 'min' in request.json:
        min_res = request.json['min']#前端要求至少显示的结果数
        if min_res is None: min_res = settings['mincandicates']
        else: min_res = request.json['min']

    threshold = settings['rela']
    print('img retrieval thres = ' + str(threshold))
    query = request.json['query']#目标图片的相对路径
    print('query img '+query +' min= '+str(min_res)+' thres = ' + str(threshold))

    if 'pos' in request.json:
        print('search partitial')
        img = Image.open(query)
        box = request.json['pos']
        size = img.size
        w, h = size
        box[0] = int(box[0] * w)
        box[1] = int(box[1] * h)
        box[2] = int(box[2] * w)
        box[3] = int(box[3] * h)
        print(box)
        img = img.crop(box)
        query = 'temp/query_crop.png'
        img.save(query)
        qvec = extract_feat(query)
        scores = corner_retrieval(qvec)
    else:
        qvec = extract_feat(query)
        scores = np.dot(qvec,np.asarray(feats).T)

    ranked_idx = np.argsort(scores)[::-1]

    candicates,num = [],0
    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()
    for i in range(max_res):
        id = ranked_idx[i]
        webpath = names[id]
        score = scores[id]
        if score<threshold and i>=int(min_res)+1: break
        relpath = relpath_from_webpath(webpath)
        im = {'id': relpath,
              'index': num,
              'webpath': webpath,
              # 'thumbnail': 'atom:///' + relpath,
              # 'original': 'atom:///' + relpath,
              'thumbnail': HOST + thumbnail_from_webpath(webpath),
              'original': HOST + webpath,
              'score':score,
              'details': get_img_detail(webpath,cursor),
              'tags': get_tag(webpath)}
        if(im['id']==query): continue
        candicates.append(im)#去掉同一图片
        num += 1
    # if(relpath_from_webpath(candicates[0][0])==query): candicates = candicates[1:]
    detect.close()
    return {'num':num,'candicates':candicates}

def corner_retrieval(qvec):
    cfeats = np.asarray(cornerfeats)
    scores4 = []
    for i in range(4):
        scores = np.dot(qvec, np.asarray(cfeats[:,i]).T)
        scores4.append(scores)
    scores4.append(np.dot(qvec, np.asarray(feats).T))
    scores4 = np.asarray(scores4)
    newscores = []
    for i in range(len(cfeats)):
        max_score = np.max(scores4[:,i])
        newscores.append(max_score)
    return newscores


#获取指定图片的清晰度
def get_fm(webpath):
    blur_res = sqlite3.connect(database_file_path)
    cursor = blur_res.cursor()
    cursor.execute('select fm from blur where webpath = ?',(webpath,))
    fm = cursor.fetchone()[0]
    return fm

@retrievalapp.route('/cpt_all/<path:dir>',methods=['GET','POST'])
def cpt_all(dir):
    print(names)
    print(len(feats))
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
        relative = []
        return {'total': 0, 'relative': relative}
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
        if(relpath_from_webpath(names1[arg[1]]) and os.path.exists(relpath_from_webpath(names1[arg[1]]))):#该图片存在
            if filt:#指定文件夹的相似图片
                if names1[arg[1]][0:dir_len]==dir:
                    rela_imgs[cur].append((names1[arg[1]],mat[arg[0],arg[1]],get_fm(names1[arg[1]])))
                #不需要指定文件夹的相似图片
            else: rela_imgs[cur].append((names1[arg[1]],mat[arg[0],arg[1]],get_fm(names1[arg[1]])))


    #删除图片数不足两张的,组内按清晰度排序
    purifed_imgs,num = [],0
    for img_list in rela_imgs:
        length = len(img_list)
        if(length>1):
            img_list = [{
                'id':relpath_from_webpath(img_score[0]),
                'index':num+i,
                'webpath': img_score[0],
                'thumbnail': HOST + thumbnail_from_webpath(img_score[0]),
                'original':HOST + img_score[0],
                # 'thumbnail': 'atom:///' + relpath_from_webpath(img_score[0]),
                # 'original': 'atom:///' + relpath_from_webpath(img_score[0]),
                'tags':get_tag(img_score[0]),
                'checked':not i==0,
                'details': get_img_detail(img_score[0],cursor),
                'score':img_score[1],
                'fm':img_score[2]
            } for i,img_score in enumerate(img_list)]
            purifed_imgs.append(img_list)
            num+=length
    detect.close()
    return {'total':num,'relative':purifed_imgs}
