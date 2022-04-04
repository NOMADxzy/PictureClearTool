import time

from flask import Blueprint,request
import os
import sys
from pathlib import Path
from tools.yolo import pre_single,draw_box
from tools.general import PathDict,relpath_from_webpath,thumbnail_from_webpath,HOST,get_tag
from detection.blur_detect import run_blur_detect,CachedBlurImg
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, colorstr,
                           increment_path, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
#目标检测相关的api在这里
detectapp = Blueprint('img_detect',__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}#判断格式正确
def isallowed_ext(s):
    return '.' in s and s.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


names= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

@detectapp.route('/',methods=['POST'])
def detect():
    relpath = request.json['source']
    print(relpath)
    if(isallowed_ext(relpath)):
        pre_res = pre_single(source=relpath)
    img0 = cv2.imread(relpath)
    img1 = draw_box(img0,pre_res)
    path = 'temp/'+str(time.time())+'_'+relpath.rsplit('/',1)[1]#放临时文件夹下
    cv2.imwrite(path,img1)
    return ({'pre_res': pre_res,'box_img':path})

@detectapp.route('/blur_detect',methods=['GET'])
def blur_detect():
    blur_imgs,num = [],0
    if(not CachedBlurImg==[]):
        for webpath,ft in CachedBlurImg:
            img = {
                 'id': relpath_from_webpath(webpath),
                 'index': num,
                 'thumbnail': HOST + thumbnail_from_webpath(webpath),
                 'original':HOST + webpath,
                 # 'webformatURL': HOST+'data/images/'+'IMG20170819123559.jpg',
                 'tags': get_tag(None,webpath),
                 'ft':ft
            }
            blur_imgs.append(img)
    else:
        for webdir in PathDict:
            blur_webpaths = run_blur_detect(webdir)
            for webpath,ft in blur_webpaths:
                img = {
                     'id': relpath_from_webpath(webpath),
                     'index': num,
                     'thumbnail': HOST + thumbnail_from_webpath(webpath),
                     'original':HOST + webpath,
                     # 'webformatURL': HOST+'data/images/'+'IMG20170819123559.jpg',
                     'tags': get_tag(None,webpath),
                     'ft':ft
                }
                blur_imgs.append(img)
    return {'blur_imgs':blur_imgs}

# #加载模型
# device = select_device('')
# model = DetectMultiBackend('weights/final.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
# @torch.no_grad()
# def pred(
#         source='data/temp/bus.jpg',  # file/dir/URL/glob, 0 for webcam
#         imgsz=(640, 640),  # inference size (height, width)
#         conf_thres=0.25,  # confidence threshold
#         iou_thres=0.45,  # NMS IOU threshold
#         max_det=1000,  # maximum detections per image
#         save_txt=False,  # save results to *.txt
#         save_crop=False,  # save cropped prediction boxes
#         nosave=False,  # do not save images/videos
#         visualize=False,  # visualize features
#         project=ROOT / 'runs/detect',  # save results to project/name
#         name='exp',  # save results to project/name
#         exist_ok=False,  # existing project/name ok, do not increment
#         ):
#     boxs = [] #存放检测结果
#     source = str(source)
#     save_img = not nosave and not source.endswith('.txt')  # save inference images
#     is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
#     is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
#     webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
#     if is_url and is_file:
#         source = check_file(source)  # download
#
#     # Directories
#     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
#
#
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = check_img_size(imgsz, s=stride)  # check image size
#
#
#     # Dataloader
#     if webcam:
#         view_img = check_imshow()
#         cudnn.benchmark = True  # set True to speed up constant image size inference
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
#         bs = len(dataset)  # batch_size
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
#         bs = 1  # batch_size
#     vid_path, vid_writer = [None] * bs, [None] * bs
#
#     # Run inference
#     model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
#     dt, seen = [0.0, 0.0, 0.0], 0
#     for path, im, im0s, vid_cap, s in dataset:
#         t1 = time_sync()
#         im = torch.from_numpy(im).to(device)
#         im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#         im /= 255  # 0 - 255 to 0.0 - 1.0
#         if len(im.shape) == 3:
#             im = im[None]  # expand for batch dim
#         t2 = time_sync()
#         dt[0] += t2 - t1
#
#         # Inference
#         visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#         pred = model(im, augment=False, visualize=visualize)
#         t3 = time_sync()
#         dt[1] += t3 - t2
#
#         # NMS
#         pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)
#         dt[2] += time_sync() - t3
#
#         # Second-stage classifier (optional)
#         # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
#
#         # Process predictions
#         for i, det in enumerate(pred):  # per image
#             seen += 1
#             if webcam:  # batch_size >= 1
#                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
#                 s += f'{i}: '
#             else:
#                 p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
#
#             p = Path(p)  # to Path
#             save_path = 'temp/'+str(time.time())+p.name  # im.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
#             s += '%gx%g ' % im.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if save_crop else im0  # for save_crop
#
#
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
#
#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
#
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det.tolist()):
#                     box = (cls, *xyxy, conf)   # label format
#                     boxs.append(box)
#
#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     cv2.imwrite(save_path, draw_box(im0,boxs))
#
#         # Print time (inference-only)
#         LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
#         return boxs,save_path
#
#     # Print results
#     t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
#     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
# def draw_box(img,boxs):
#     annotator = Annotator(img, line_width=3, example=str(names))
#     for cls,*xyxy,conf in boxs:
#         c = int(cls)  # integer class
#         label = f'{names[c]} {conf:.2f}'
#         annotator.box_label(xyxy, label, color=colors(c, True))
#     return annotator.result()
# def test_draw():
#     img = cv2.imread('/Users/macos/Desktop/yolov5-master/data/temp/bus.jpg')
#     boxs = [[0,0,562,81,882,0.308542]]
#     img1 = draw_box(img,boxs)
#     cv2.imshow("1", img1)
#     cv2.waitKey()