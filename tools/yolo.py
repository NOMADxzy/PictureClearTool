import time
import sqlite3,pickle
from flask import Blueprint,request
import os
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

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}#判断格式正确
def is_allowed_ext(s):
    return '.' in s and s.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#加载模型
device = select_device('')
model = DetectMultiBackend('weights/final.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)

@torch.no_grad()
def pre_single(
        source='data/temp/bus.jpg',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        ):
    boxs = [] #存放检测结果
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
                    box = (cls, *xyxy, conf)   # label format
                    boxs.append(box)


        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        return boxs

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

def draw_box(img,boxs):
    annotator = Annotator(img, line_width=3, example=str(names))
    for cls,*xyxy,conf in boxs:
        c = int(cls)  # integer class
        label = f'{names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))
    return annotator.result()
names= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
def pre_dir(web_dir,dir):#对文件夹中的所有图片检测出box,根据tag分类，写入数据库
    cluster = {}
    detect = sqlite3.connect("detect_results.db")
    cursor = detect.cursor()
    for root, dirs, files in os.walk(dir):
        dirs[:] = []
        for file in files:
            if (not is_allowed_ext(file)): continue

            web_path = web_dir+'/'+file
            rel_path = root + '/' + file
            boxs = pre_single(source=rel_path)
            for box in boxs:
                cls = int(box[0])
                if(cls in cluster):
                    if(web_path not in cluster[cls]): cluster[cls].append(web_path)#防止一张图片多次添加到同一个标签下
                else:
                    cluster[cls] = [web_path]
            tags = [names[int(box[0])] for box in boxs]
            tags_dump = pickle.dumps(set(tags))
            boxs_dump = pickle.dumps(boxs)
            cursor.execute("""select * from Tag where path=(?)""",(web_path,))
            exist = cursor.fetchone()
            if(exist==None):
                cursor.execute("""insert into Tag values (?,?,?)""",(web_path,boxs_dump,tags_dump))
    # 写入tag聚类表
    for cls in cluster:
        cursor.execute("""select * from tagclassified where tag=(?)""", (cls,))
        res = cursor.fetchone()
        if (res == None):
            imgs_dump = pickle.dumps(list(set(cluster[cls])))
            cursor.execute("""insert into tagclassified values (?,?)""", (cls, imgs_dump))
        else:
            cls, img0s_dump = res
            img0s = pickle.loads(img0s_dump)
            img1s_dump = pickle.dumps(list(set(img0s + cluster[cls])))
            cursor.execute("""update tagclassified set imgs = ? where tag = ?;""", (img1s_dump, cls))

    detect.commit()
    detect.close()
