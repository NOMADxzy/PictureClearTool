import os
from PIL import Image
import pickle,sqlite3

PathDict = {'pics':'pics','Desktop':'..','test_blur':'test_blur'}
HOST = 'http://localhost:5000/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','webp'}#判断格式正确
def is_allowed_ext(s):
    return '.' in s and s.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

def get_img_paths(path):
    imglist = []
    for root, dirs, files in os.walk(path):
        dirs[:] = []
        for file in files:
            if (not is_allowed_ext(file)): continue
            imglist.append(root+'/'+file)
    return imglist

def get_thumbnail_pic(path):#生成单张图片或文件夹下所有图片的缩略图
    if(os.path.isfile(path)):
        img = Image.open(path)
        path_splited = path.rsplit('/',1)
        thumb_path = path_splited[0] + '/.thumbnail/'
        if (not os.path.exists(thumb_path)): os.mkdir(thumb_path)
        img.thumbnail((300, 300))
        img.save(thumb_path+path_splited[1], 'PNG')
    else:
        for root, dirs, files in os.walk(path):
            dirs[:] = []
            for file in files:
                if (not is_allowed_ext(file)): continue
                name = root + '/' + file
                img = Image.open(name)
                thumb_path = root + '/.thumbnail/'
                if (not os.path.exists(thumb_path)): os.mkdir(thumb_path)
                name = thumb_path + file
                img.thumbnail((300, 300))
                img.save(name, 'PNG')

def relpath_from_webpath(webpath):#webpath 转成系统相对路径relpath
    pathsplited = webpath.rsplit('/',1)
    if(pathsplited[0] not in PathDict): return False
    else:
        return PathDict[pathsplited[0]]+'/'+ pathsplited[1]

def thumbnail_from_webpath(webpath):
    webpathsplited = webpath.rsplit('/',1)
    return webpathsplited[0]+'/.thumbnail/'+webpathsplited[1]

def get_tag(cursor,webpath):
    inside_sql = False#方法内部连接数据库
    if(cursor==None):
        inside_sql = True
        detect = sqlite3.connect("detect_results.db")  # 连接数据库
        cursor = detect.cursor()
    cursor.execute("""select * from Tag where path=(?)""", (webpath,))
    result = cursor.fetchone()
    if (result == None):
        tags = []
    else:
        p,boxs_dump,tags_dump = result
        tags = list(pickle.loads(tags_dump))
    if(inside_sql):
        detect.commit()
        detect.close()
    return tags