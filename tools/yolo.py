import time
import sqlite3, pickle
from flask import Blueprint, request
import os,time
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, colorstr,
                           increment_path, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
from tools.general import names,TagGroup,relpath_from_webpath,Tag,\
    PathDict,is_allowed_ext,settings
from tools.val import database_file_path,yolo_weights_paths,cls_idx_base

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
        rewrite = False
        for img in imgs:
            if (not (relpath_from_webpath(img) and os.path.exists(relpath_from_webpath(img)) and img.split('/')[
                0] in PathDict)):
                print('yolo ' + img + ' can not find (Tag Group)')
                imgs.remove(img)
                rewrite = True
        # 图片组发生改变则重写
        if (rewrite):
            print('(yolo)rewrite tag group ' + names[tag])
            cursor.execute("""update TagGroupTable set imgs = ? where tag = ?;""", (pickle.dumps(imgs), tag))
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
if(settings['weight']=='高'):
    weight_path = yolo_weights_paths[0]
elif settings['weight']=='中':
    weight_path = yolo_weights_paths[1]
else:
    weight_path = yolo_weights_paths[2]
device = select_device('')
model = DetectMultiBackend(weight_path, device=device, dnn=False, data='data/coco128.yaml',
                                    fp16=False)
mymodel = False
if(os.path.exists('weights/best.pt')):
    mymodel = DetectMultiBackend('weights/best.pt', device=device, dnn=False, data='data/my_class_train100.yaml',
                                    fp16=False)

def pre_boxs(relpath):
    boxs1 = pre_single(model=model,source=relpath)
    #第二个模型存在就再后面加上第二个模型的结果
    if mymodel:
        boxs2 = pre_single(model=mymodel, source=relpath)
        for box in boxs2:
            box[0] = box[0] + cls_idx_base
            boxs1.append(box)
    return boxs1

# 加载
@torch.no_grad()
def pre_single(model=model,
        source='data/temp/bus.jpg',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.45,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
):
    boxs = []  # 存放检测结果
    source = str(source)

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det.tolist()):
                    box = [cls, *xyxy, conf]  # label format
                    boxs.append(box)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        return boxs


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
            tags = [names[int(box[0])] for box in boxs]
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
            print('(yolo) updating tag group '+names[cls]+' add'+str(cluster[cls]))
            cursor.execute("""update TagGroupTable set imgs = ? where tag = ?;""", (img1s_dump, cls))
            TagGroup[cls] = list(set(img0s + cluster[cls]))

    detect.commit()
    detect.close()
    return total

#扫描所有文件夹，检测新增的图片
new_num = 0
for web_dir in PathDict:
    new_num += pre_dir(web_dir)
t4 = time.process_time()
print("(yolo)check and detect new images,"+str(new_num)+" done spent time: "+str(t4-t3))




def draw_box(img, boxs):
    annotator = Annotator(img, line_width=10, example=str(names))
    for cls, *xyxy, conf in boxs:
        c = int(cls)  # integer class
        label = f'{names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))
    return annotator.result()
