from flask import Flask, request, make_response
from pathlib import Path
from tools.general import is_allowed_ext,get_thumbnail_pic,get_tag,\
    HOST,PathDict,TagGroup,Tag,names,webpath_from_relpath

import os,pickle,sqlite3
from flask_cors import CORS

from retrieval.img_retrieval import retrievalapp
from detects.img_detect import detectapp
app = Flask(__name__)
CORS(app,supports_credentials=True)
app.register_blueprint(retrievalapp,url_prefix='/retrieval')
app.register_blueprint(detectapp,url_prefix='/detect')





#获取、查看、删除图片等基本接口在这里

#获取指定文件夹下的图片
@app.route('/get_pics/<path:webdir>',methods=['GET'])
def get_pics(webdir,baseindex=0):
    if not webdir in PathDict:
        print('dir not exist')
        return {'total':0,'imgs':[]}
    root = PathDict[webdir]
    baseindex#用于get_all_pics的index矫正

    detect = sqlite3.connect("detect_results.db")#连接数据库
    cursor = detect.cursor()
    for root, dirs, files in os.walk(str(root)):
        hits,dirs[:],num=[],[],0#忽略子文件夹
        for file in files:
            if (not is_allowed_ext(file)): continue

            thumb_rel_path = root + '/.thumbnail/' + file
            if(not os.path.exists(thumb_rel_path)):#缩略图不存在,生成缩略图
                get_thumbnail_pic(root+'/'+file)

            tag = get_tag(webdir+'/'+file)
            img = {'id':root+'/'+file,
                   'index':num+baseindex,
                   'thumbnail':HOST + webdir + '/.thumbnail/' + file,
                   'original':HOST+webdir+'/'+file,
                   # 'webformatURL': HOST+'data/images/'+'IMG20170819123559.jpg',
                   'tags':tag}
            hits.append(img)
            num += 1
    detect.commit()
    detect.close()
    return ({'total':num,'imgs':hits})  # 当前路径下所有非目录子文件

@app.route('/get_all_pics',methods=['GET'])
@app.route('/search/',methods=['GET'])
def get_all_pics():
    total = 0
    imgs = []
    for web_dir in PathDict:
        res = get_pics(web_dir,len(imgs))
        total += res['total']
        imgs += res['imgs']
    return {'total':total,'imgs':imgs}

@app.route('/<path:dir>/<path:file>', methods=['GET'])
def show_photo(dir,file):
    if not file is None:
        if  dir in PathDict or dir == 'temp':
            if dir=='temp': root='temp'
            else: root = PathDict[dir]
            image_data = open(f'{root}/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
        else:
            print('file not exist')
            return 'none', 200

@app.route('/search/<path:tag_name>',methods=['GET'])
def search(tag_name):
    print(tag_name)
    if(tag_name==''): return get_all_pics() #返回所有图片
    # detect = sqlite3.connect("detect_results.db")
    # cursor = detect.cursor()
    # if(not tag_name in names):#没有该标签
    #     return {'imgs':[]}
    # tag = names.index(tag_name)
    # cursor.execute("""select * from tagclassified where tag = ?""",(tag,))
    # result = cursor.fetchone()
    # if(result==None):#没有记录
    #     return {'imgs':[]}
    # t,imgs_dump = result
    # imgs = pickle.loads(imgs_dump)
    imgs = TagGroup[names.index(tag_name)]

    imgs_pack = []
    num = 0
    for img in imgs:
        img_splited = img.rsplit('/',1)
        root = PathDict[img_splited[0]]
        #移除失效路径
        if (img_splited[0] not in PathDict or not os.path.exists(
            root+ '/' + img_splited[1])): imgs.remove(img)

        thumb_img = img_splited[0]+'/.thumbnail/'+img_splited[1]
        im = {'id': root+'/'+img_splited[1],
              'index': num,
              'thumbnail': HOST + thumb_img,
              'original': HOST + img,
              'tags': get_tag(img)}
        imgs_pack.append(im)
        num+=1
    return {'total':num,'imgs':imgs_pack}

@app.route('/delete',methods=['POST'])
def delete():
    detect = sqlite3.connect("detect_results.db")
    cursor = detect.cursor()
    relpaths = request.json['paths']
    paths = [webpath_from_relpath(relpath) for relpath in relpaths]
    print('del '+str(paths))
    # cursor.execute("""select * from tagclassified""")
    # result_list = cursor.fetchall()
    # result_dict = {tag:pickle.loads(img0s_dump) for tag,img0s_dump in result_list}
    for path in paths:
        if(not os.path.exists(path)): continue
        if(not path in Tag):continue
        tag_names = get_tag(path)

        tags = [names.index(tag_name) for tag_name in tag_names]#转成序号
        for tag in tags:
            TagGroup[tag].remove(path)
        os.remove(path)  # 文件删除
        Tag.pop(path) #Tag表中删除
    print('current tag group size: '+str(len(TagGroup[0])))
    if(len(paths)>0):
        s = str(paths)[1:-1]
        cursor.execute("delete from Tag where path in (" + s + ")")
    for tag in TagGroup:
        cursor.execute("""update tagclassified set imgs = ? where tag = ?""", (pickle.dumps(TagGroup[tag]),tag))
    detect.commit()
    detect.close()
    return 'done',200

@app.route('/get_dirs',methods=['GET'])
def get_dirs():
    return {'dirs':list(PathDict.keys())}

@app.route('/import_dir',methods=['POST'])
def import_dir():
    dir = request.json['dir']
    print(dir)
    dir_splited = dir.rsplit('/',1)[1]
    if(len(dir_splited)==2):
        web_dir = dir_splited[1]
    if(web_dir=='temp'): return 'temp reserved',202
    orig_dir = web_dir
    times = 1
    while(web_dir in PathDict):#后面加数字与已出现的同名文件夹区分
        web_dir = orig_dir + str(times)
        times+=1

    print(web_dir)
    rel_path = os.path.relpath(dir,Path.cwd())
    print(rel_path)
    PathDict[web_dir] = rel_path
    # pre_dir(web_dir, rel_path)
    return {'webdir':web_dir}
@app.route('/del_dir',methods=['POST'])
def del_dir():
    dir = dir = request.json['dir']


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

