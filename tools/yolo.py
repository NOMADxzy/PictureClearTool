import sqlite3, pickle,os,time


from tools.general import names,TagGroup,relpath_from_webpath,Tag,\
    PathDict,is_allowed_ext,settings
from tools.val import database_file_path,yolo_weights_paths,cls_idx_base

from remotes.RPC import pre_boxs

# 加载标签分类信息(TagGroupTable),清理不存在的图片
t1 = time.process_time()
detect = sqlite3.connect(database_file_path)
cursor = detect.cursor()
for tag in range(len(names)):  # 读取TagGroupTable表
    cursor.execute("""select * from TagGroupTable where tag = ?""", (tag,))
    result = cursor.fetchone()
    if (result == None):  # 没有记录
        TagGroup[tag] = []
        cursor.execute("""insert into TagGroupTable values (?,?);""", (tag,pickle.dumps([])))
    else:
        t, imgs_dump = result
        imgs = pickle.loads(imgs_dump)
        imgs1 = []

        for img in imgs:
            if not relpath_from_webpath(img):
                print('yolo ' + img + ' can not find (Tag Group)')
            else:
                imgs1.append(img)

        # 图片组发生改变则重写
        if len(imgs) != len(imgs1):
            print('(yolo)rewrite tag group ' + names[tag])
            cursor.execute("""update TagGroupTable set imgs = ? where tag = ?;""", (pickle.dumps(imgs1), tag))
        TagGroup[tag] = imgs  # 读取到内存中使用
t2 = time.process_time()
print("(yolo)load and check tag group, done spent time: " + str(t2 - t1))
# 加载box检测结果（Tag）
cursor.execute("""select * from TagTable""")
result = cursor.fetchall()
if (not result == None):  # 有数据
    rewrite, del_list = False, []
    for r in result:
        webpath, box_dump, tag_dump, de = r
        if (not (relpath_from_webpath(webpath) and os.path.exists(relpath_from_webpath(webpath)) and
                 webpath.split('/')[0] in PathDict)):
            print(webpath + '(yolo) can not find (Tag)')
            rewrite = True
            del_list.append(webpath)
        else:
            Tag[webpath] = (pickle.loads(box_dump), pickle.loads(tag_dump))
    if (rewrite):
        cursor.execute("delete from TagTable where path in (" + str(del_list)[1:-1] + ")")
        print('(yolo) rewrite tag delete' + str(del_list))

t3 = time.process_time()
print("(yolo)load and check Tag, done spent time: " + str(t3 - t2))
detect.commit()
detect.close()

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# 加载模型

def pre_dir(web_dir):  # 对文件夹中的所有图片检测出box,根据tag分类，写入数据库
    if(not os.path.isdir(PathDict[web_dir])):
        print('(yolo) ' + web_dir + ' 文件夹不存在')
        return 0
    cluster = {}
    detect = sqlite3.connect(database_file_path)
    cursor = detect.cursor()
    total = 0 #new image num
    for root, dirs, files in os.walk(PathDict[web_dir]):
        dirs[:] = []
        for file in files:
            if (not is_allowed_ext(file)): continue

            web_path = web_dir + '/' + file
            rel_path = root + '/' + file
            if(web_path in Tag): continue#有就不预测了
            total += 1
            print('(yolo) detecting tag from '+web_path)
            boxs = pre_boxs(rel_path)
            for box in boxs:
                cls = int(box[0])
                if (cls in cluster):#添加到相应的聚类中
                    if (web_path not in cluster[cls]): cluster[cls].append(web_path)  # 防止一张图片多次添加到同一个标签下
                else:
                    cluster[cls] = [web_path]
            tags = []
            for box in boxs:
                if int(box[0])<len(names):
                    tags.append(names[int(box[0])])

            Tag[web_path] = [(boxs,tags)]
            tags_dump = pickle.dumps(list(set(tags)))
            boxs_dump = pickle.dumps(boxs)
            cursor.execute("""select * from TagTable where path=(?)""", (web_path,))
            exist = cursor.fetchone()
            if (exist == None):
                Tag[web_path] = (boxs,list(set(tags)))
                cursor.execute("""insert into TagTable values (?,?,?,?)""", (web_path, boxs_dump, tags_dump,0))
    # 写入tag聚类表
    for cls in cluster:
        cursor.execute("""select * from TagGroupTable where tag=(?)""", (cls,))
        res = cursor.fetchone()
        if (res == None):
            imgs_dump = pickle.dumps(list(set(cluster[cls])))
            print('(yolo) creating tag group' + names[cls] + ' add' + str(cluster[cls]))
            cursor.execute("""insert into TagGroupTable values (?,?)""", (cls, imgs_dump))
            TagGroup[cls] = list(set(cluster[cls]))
        else:
            cls, img0s_dump = res
            img0s = pickle.loads(img0s_dump)
            img1s_dump = pickle.dumps(list(set(img0s + cluster[cls])))
            if cls>len(names)-1: break #忽略超出names的group
            print('(yolo) updating tag group '+names[cls]+' add'+str(cluster[cls]))
            cursor.execute("""update TagGroupTable set imgs = ? where tag = ?;""", (img1s_dump, cls))
            TagGroup[cls] = list(set(img0s + cluster[cls]))

    detect.commit()
    detect.close()
    return total


# def draw_box(img, boxs):
#     annotator = Annotator(img, line_width=10, example=str(names))
#     for cls, *xyxy, conf in boxs:
#         c = int(cls)  # integer class
#         label = f'{names[c]} {conf:.2f}'
#         annotator.box_label(xyxy, label, color=colors(c, True))
#     return annotator.result()
