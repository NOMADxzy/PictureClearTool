# coding=UTF-8
from flask import Flask, request, make_response,redirect
from pathlib import Path
from tools.general import is_allowed_ext, get_thumbnail_pic, get_tag,settings, \
    HOST, PathDict, TagGroup, Tag, names, webpath_from_relpath, get_img_detail,get_img_paths,relpath_from_webpath
from tools.val import database_file_path,pathdict_file_path,settings_file_path
import os, pickle, sqlite3
from flask_cors import CORS

from retrieval.img_retrieval import retrievalapp
from detects.img_detect import detectapp

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.register_blueprint(retrievalapp, url_prefix='/retrieval')
app.register_blueprint(detectapp, url_prefix='/detect')

# 初始，检查已注册的文件夹
for dir in PathDict:
    reldir = PathDict[dir]
    if not os.path.exists(reldir):
        print(dir + 'not exist anymore')
        PathDict.pop(dir)


# 获取、查看、删除图片等基本接口在这里
@app.route('/', methods=['GET'])
def hello():
    return {'name':'image tool backend',
            'pathdict':PathDict,'taggroup':TagGroup,'tag':Tag}


# 获取指定文件夹下的图片
@app.route('/get_pics/<path:webdir>', methods=['GET'])
def get_pics(webdir, baseindex=0):
    if(webdir=='') : return redirect('/get_all_pics')
    if not webdir in PathDict or not os.path.isdir(PathDict[webdir]):
        print(webdir + '(get_pics) dir not exist')
        imgs = []
        return {'total': 0, 'imgs': imgs}
    root = PathDict[webdir]
    print('get pics from '+root)
    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()
    # baseindex#用于get_all_pics的index矫正

    for root, dirs, files in os.walk(str(root)):
        hits, dirs[:], num = [], [], 0  # 忽略子文件夹
        for file in files:
            if (not is_allowed_ext(file)): continue

            thumb_rel_path = root + '/.thumbnail/' + file
            if (not os.path.exists(thumb_rel_path)):  # 缩略图不存在,生成缩略图
                get_thumbnail_pic(root + '/' + file)

            tag = get_tag(webdir + '/' + file)
            img = {'id': root + '/' + file,
                   'index': num + baseindex,
                   'thumbnail': HOST + webdir + '/.thumbnail/' + file,
                   # 'thumbnail': 'atom:///'+root+'/.thumbnail/'+file,
                   # 'original': 'atom:///'+root+'/'+file,
                   'original':HOST + webdir + '/' + file,
                   'details': get_img_detail(root + '/' + file,cursor),
                   # 'webformatURL': HOST+'data/images/'+'IMG20170819123559.jpg',
                   'tags': tag}
            hits.append(img)
            num += 1
    detect.commit()
    detect.close()
    return ({'total': num, 'imgs': hits})  # 当前路径下所有非目录子文件


@app.route('/get_all_pics', methods=['GET'])
@app.route('/search/', methods=['GET'])
def get_all_pics():
    total = 0
    imgs = []
    imgsdict = {}
    for web_dir in PathDict:
        res = get_pics(web_dir, len(imgs))
        total += res['total']
        imgs += res['imgs']
        imgsdict[web_dir] = res['imgs']
    return {'total': total, 'imgs': imgs,'imgsdict':imgsdict}

@app.route('/<path:dir>/<path:file>', methods=['GET'])
def show_photo(dir, file):
    if not file is None:
        if dir in PathDict or dir == 'temp':
            if dir == 'temp':
                root = 'temp'
            elif dir == 'temp/avatar':
                root = 'temp/avatar'
            else:
                root = PathDict[dir]
            image_data = open(f'{root}/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
        else:
            print('file not exist')
            return 'none', 200


@app.route('/delete', methods=['POST'])
def deletefiles():
    webpaths = list(set(request.json['paths']))
    print('remove '+str(webpaths))
    deletetags(webpaths,osremove=True)
    return str(len(webpaths)), 200

def deletetags(webpaths,osremove=False):
    detect = sqlite3.connect(database_file_path)
    cursor = detect.cursor()
    paths = webpaths
    # cursor.execute("""select * from tagclassified""")
    # result_list = cursor.fetchall()
    # result_dict = {tag:pickle.loads(img0s_dump) for tag,img0s_dump in result_list}
    for path in paths:
        relpath = relpath_from_webpath(path)
        if (not os.path.exists(relpath)): continue

        tag_names = get_tag(path)

        tags = [names.index(tag_name) for tag_name in tag_names]  # 转成序号
        for tag in tags:
            if(path in TagGroup[tag]):TagGroup[tag].remove(path)
        if osremove: os.remove(relpath)  # 文件删除
        if (path in Tag): Tag.pop(path)  # Tag表中删除
    print('current tag group size: ' + str(len(TagGroup[0])))
    if (len(paths) > 0):
        s = str(paths)[1:-1]
        cursor.execute("delete from TagTable where path in (" + s + ")")
    for tag in TagGroup:
        cursor.execute("""update TagGroupTable set imgs = ? where tag = ?""", (pickle.dumps(TagGroup[tag]), tag))
    detect.commit()
    detect.close()


@app.route('/get_dirs', methods=['GET'])
def get_dirs():
    return {'dirs': list(PathDict.keys())}




@app.route('/del_dir', methods=['GET'])
def del_dir():
    # dir = request.json['dir']
    dir = request.args['dir']
    if(dir not in PathDict): return 'not exist',202
    paths = get_img_paths(dir=PathDict[dir],webpath=dir)#获取所有的webpaths
    deletetags(paths,osremove=False)
    # os.remove(PathDict[dir] + '/.thumbnails')#删除缩略图缓存
    PathDict.pop(dir)
    with open(pathdict_file_path,'wb') as file:
        pickle.dump(PathDict,file)
        file.close()
    return 'done',200

@app.route('/setting', methods=['POST','GET'])
def setting():
    if request.method=='GET':
        return settings
    print(request.json)
    type = request.json['type']
    val = request.json['val']
    # if not type in settings: return 'error',400
    settings[type] = val

    with open(settings_file_path, 'wb') as file:
        pickle.dump(settings,file)
        file.close()
    return 'ok',200


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
