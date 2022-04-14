import os
from PIL import Image
import pickle,sqlite3,re
import time

PathDict = {'pics':'pics','Desktop':'..','test_blur':'test_blur','person':'person','foods':'../../Pictures/foods'}
with open('PathDict.pkl','rb') as file:
    PathDict = pickle.load(file)
    file.close()
PathDict['pics'] = 'pics'
HOST = 'http://localhost:5000/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','webp'}#判断格式正确
names= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
Tag,TagGroup = {},{}
#Tag:{webpath:[boxs,tags]...}
#TagGroup:{tag:[webpath1,webpath2...],[...]...}
def is_allowed_ext(s):
    if s[0]== '.': return False
    return '.' in s and s.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

def is_screen_shot(relpath):
    file = relpath.rsplit('/',1)[1]
    details = get_img_detail(relpath)
    general_resolution = [(1080,1920),(720,1280),(1080,2400)]
    return file[0:10].lower()=='screenshot' or details[2] in general_resolution

def get_img_paths(dir,webpath=False):#由relpdir获取relpaths
    imglist = []
    for root, dirs, files in os.walk(dir):
        dirs[:] = []
        for file in files:
            if (not is_allowed_ext(file)): continue
            if(webpath):
                imglist.append(webpath+'/'+file)
            else:
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
    if(pathsplited[0] not in PathDict):
        return False
    else:
        return PathDict[pathsplited[0]]+'/'+ pathsplited[1]
def webpath_from_relpath(relpath):#relpath 转成网络路径webpath
    pathsplited = relpath.rsplit('/',1)
    for webdir in PathDict:
        if(PathDict[webdir]==pathsplited[0]): return webdir+'/'+pathsplited[1]
    else: return False


def thumbnail_from_webpath(webpath):
    webpathsplited = webpath.rsplit('/',1)
    return webpathsplited[0]+'/.thumbnail/'+webpathsplited[1]

def get_tag(webpath):
    return list(Tag[webpath][1]) if(webpath in Tag) else []

def webpath_belongto_dir(webpath,dir):
    pathsplited = webpath.rsplit('/', 1)
    return webpath[0:len(dir)] == dir

def get_img_detail(relpath):
    import time
    img = Image.open(relpath)
    info = img._getexif()
    size = os.path.getsize(relpath)

    if info is None or 306 not in info:
        time = time.strftime("%Y:%m:%d %H:%M:%S", time.localtime(os.stat(relpath).st_mtime))
        date_and_time = time.split(' ')  # 只要日期
        return [size,date_and_time[0],img.size,date_and_time[1]]
    else:
        time = info[306]
        date_and_time = time.split(' ')#日期和时间分开

    return [size,date_and_time[0], img.size,date_and_time[1]]


    # inside_sql = False#方法内部连接数据库
    # if(cursor==None):
    #     inside_sql = True
    #     detect = sqlite3.connect("detect_results.db")  # 连接数据库
    #     cursor = detect.cursor()
    # cursor.execute("""select * from Tag where path=(?)""", (webpath,))
    # result = cursor.fetchone()
    # if (result == None):
    #     tags = []
    # else:
    #     p,boxs_dump,tags_dump = result
    #     tags = list(pickle.loads(tags_dump))
    # if(inside_sql):
    #     detect.commit()
    #     detect.close()
    # return tags
