import copy
import pickle,sqlite3
import time,yaml
from PIL import Image, ImageDraw
import numpy as np
from flask import Blueprint,request,redirect
import os,math
import sys
from pathlib import Path
from tools.yolo import pre_boxs,draw_box,pre_dir
from tools.general import PathDict,relpath_from_webpath,webpath_from_relpath,\
    thumbnail_from_webpath,HOST,get_tag,names,Tag,get_img_paths,is_screen_shot,webpath_belongto_dir,\
    TagGroup,get_img_detail,is_allowed_ext,executor,settings,info
from tools.val import database_file_path,pathdict_file_path,cls_idx_base
from detects.blur import compute_blur,CachedBlurImg,run_blur_detect
import cv2
from detects.face import known_face_names,known_face_imgs,get_paths ,avatars,find,generate_avatar
from detects.ocr import read
from tools.train_prepair import add_to_train


#目标检测相关的api在这里
detectapp = Blueprint('img_detect',__name__)



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



@detectapp.route('/',methods=['POST'])
def detect():
    relpath = request.json['source']
    print(relpath)
    if(is_allowed_ext(relpath)):
        pre_res = pre_boxs(source=relpath)
    img0 = cv2.imread(relpath)
    img1 = draw_box(img0,pre_res)
    path = 'temp/'+str(time.time())+'_'+relpath.rsplit('/',1)[1]#放临时文件夹下
    cv2.imwrite(path,img1)
    return ({'pre_res': pre_res,'box_img':path})

@detectapp.route('/search/',methods=['GET'])
def redirect_to_all():
    print('redirect')
    return redirect('/get_all_pics')

@detectapp.route('/search/<path:tag_name>',methods=['GET'])
def search(tag_name):
    if(tag_name not in names):
        imgs = []
        return {'total':0,'imgs':imgs}
    filt = False  # 按文件夹过滤
    if('dir' in request.args):
        dir = request.args['dir']
        if (dir in PathDict):
            filt = True

    print('searchtag '+tag_name)
    if(filt): print('searchdir: '+ dir)
    if(tag_name==''): return redirect('/get_all_pics') #返回所有图片
    imgs = TagGroup[names.index(tag_name)]

    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()

    imgs_pack = []
    num = 0
    for img in imgs:
        img_splited = img.rsplit('/',1)
        if (img_splited[0] not in PathDict): continue
        root = PathDict[img_splited[0]]
        #移除失效路径
        if (img_splited[0] not in PathDict or not os.path.exists(
            root+ '/' + img_splited[1])): imgs.remove(img)
        if(filt and not img_splited[0]==dir):
            continue

        thumb_img = img_splited[0]+'/.thumbnail/'+img_splited[1]
        im = {'id': root+'/'+img_splited[1],
              'index': num,
              'webpath': img,
              'thumbnail': HOST + thumb_img,
              'original': HOST + img,
              'details': get_img_detail(root + '/' + img_splited[1],cursor),
              'tags': get_tag(img)}
        imgs_pack.append(im)
        num+=1
    detect.close()
    return {'total':num,'imgs':imgs_pack}

@detectapp.route('/box_img',methods=['POST'])
def box_img():
    relpath = request.json['relpath']
    tag_name = request.json['tag']
    tag_index = names.index(tag_name)
    webpath = webpath_from_relpath(relpath)

    tag_boxs = [] #指定tag的boxs
    if(webpath not in Tag): return {'box_img':''}
    boxs = Tag[webpath][0]
    for box in boxs:
        if(box[0]==tag_index): tag_boxs.append(box)

    img0 = cv2.imread(relpath)
    print(tag_boxs)
    img1 = draw_box(img0,tag_boxs)
    path = 'temp/' + str(time.time()) + '_' + relpath.rsplit('/', 1)[1]  # 放临时文件夹下
    cv2.imwrite(path, img1)
    return ({'box_img': HOST+path})


@detectapp.route('/add_tag',methods=['POST'])
def add_tag():
    detect = sqlite3.connect(database_file_path)
    cursor = detect.cursor()

    webpath = request.json['webpath']
    newboxs = request.json['box']
    if(len(newboxs)==0): return
    print(newboxs)
    # add_train(webpath,newboxs)
    add_to_train(relpath_from_webpath(webpath), newboxs)
    print('add new tag ' + relpath_from_webpath(webpath))
    boxs, tags = Tag[webpath]
    mtags = []
    for box in newboxs:

        # 比例值放大成实际值
        size = Image.open(relpath_from_webpath(webpath)).size

        w, h = size
        box[1] = int(box[1] * w)
        box[2] = int(box[2] * h)
        box[3] = int(box[3] * w)
        box[4] = int(box[4] * h)

        tag = box[0]
        if tag in names:
            tagid = names.index(tag)
            if webpath not in TagGroup[tagid]: TagGroup[tagid].append(webpath)
        else:
            tagid = len(names)  # 新标签
            names.append(tag)
            cursor.execute('update Settings set value = ? where key = ?', (pickle.dumps(names), 'names'))
            #修改训练文件
            with open('./data/my_class_train100.yaml', 'w', encoding='utf8') as f:
                info['names'] = names[cls_idx_base:]
                info['nc'] = len(names) - cls_idx_base
                yaml.dump(info, f, allow_unicode=True)
                f.close()

            TagGroup[tagid] = [webpath]  # 修改内存中的

        box[0] = tagid#格式化box
        box.append(1)

        mtags.append(tagid)#返回给前端,记录改动的tagid后面修改相应的数据库
        boxs.append(box)#box一定添加
        if tag not in tags: tags.append(tag)#只有没有时才添加tag
    Tag[webpath] = (boxs, tags)#修改内存中的

    #写到数据库中
    cursor.execute("""update TagTable set boxs = ?, tag = ? where path = ?""",
                   (pickle.dumps(boxs), pickle.dumps(tags), webpath))
    for tagid in mtags:
        cursor.execute('update TagGroupTable set imgs = ? where tag = ?', (pickle.dumps(TagGroup[tagid]), tagid))

    detect.commit()
    detect.close()


    mtags_len = [len(TagGroup[tagid]) for tagid in mtags]
    return {'tag_ids':mtags,'tag_ids_len':mtags_len}

def add_train(webpath,boxs):

    relpath = relpath_from_webpath(webpath)
    norm_boxs = []
    # boxs = copy.deepcopy(Tag[webpath][0])
    w,h = Image.open(relpath_from_webpath(webpath)).size

    # for box in boxs:
    #     box[1] = box[1] / w
    #     box[2] = box[2] / h
    #     box[3] = box[3] / w
    #     box[4] = box[4] / h
    #     norm_boxs.append(box[:5])
    add_to_train(relpath,norm_boxs)



def preparedir(webdir):
    print('prepair dir')
    pre_dir(webdir)
    compute_blur(webdir)
    find(get_paths())
    generate_avatar()

@detectapp.route('/import_dir', methods=['GET'])
def import_dir():
    dir = request.args['dir']
    if(dir[-1]=='/'): dir = dir[:-1]
    print('add new directory '+dir)
    dir_splited = dir.rsplit('/', 1)
    if (len(dir_splited) == 2):
        web_dir = dir_splited[1]
    else: return 'error dir',404
    reldir = os.path.relpath(Path.cwd(), dir)
    if (web_dir == 'temp'or web_dir) in PathDict: return 'dir name already used', 400
    # orig_dir = web_dir
    # times = 1
    # while (web_dir in PathDict):  # 后面加数字与已出现的同名文件夹区分
    #     web_dir = orig_dir + str(times)
    #     times += 1

    rel_path = os.path.relpath(dir, Path.cwd())
    PathDict[web_dir] = rel_path

    total = executor.submit(preparedir,web_dir)
    redirect('retrieval/index/'+web_dir)

    with open(pathdict_file_path,'wb') as file:
        pickle.dump(PathDict,file)
        file.close()
    return web_dir

@detectapp.route('/ocr',methods=['GET'])
def ocr():
    relpath = request.args['id']
    result,float_left = read(relpath)
    return {'result':result,'float_left':float_left}

@detectapp.route('/thing',methods=['GET'])
def thing():
    thres = 1
    imgs, num =  [], 0
    g_sizes = [len(TagGroup[tagid]) for tagid in TagGroup]
    sortidxs = np.argsort(np.asarray(g_sizes))

    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()

    for iid in range(len(names)):
        id = sortidxs[len(names)-1-iid]
        if(id==0): continue#跳过人物
        if(g_sizes[id]<thres): break
        ims = []
        for i,webpath in enumerate(TagGroup[id]):
            relpath = relpath_from_webpath(webpath)
            if not relpath: continue
            im = {'id': relpath,
                  'index': num+i,
                  'webpath': webpath,
                  'thumbnail': HOST + thumbnail_from_webpath(webpath),
                  'original': HOST + webpath,
                  'name': names[id],
                  'avatar':HOST+thumbnail_from_webpath(webpath),
                  'details': get_img_detail(relpath,cursor),
                  'tags': get_tag(webpath)}
            ims.append(im)
        num += len(ims)
        if(len(ims)):imgs.append(ims)

    detect.close()
    return {'imgs':imgs,'total':num}

@detectapp.route('/face',methods=['GET'])
def face():
    personnames,imgs,num = [],[],0
    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()
    for i in range(len(known_face_names)):
        personname = known_face_names[i]
        group = []
        for j,im in enumerate(known_face_imgs[i]):
            webpath,pos = im

            relpath = relpath_from_webpath(webpath)
            if not relpath:continue
            print(len(known_face_names))
            print(len(avatars))
            img = {
                'id': relpath,
                'index': num+j,
                'webpath': webpath,
                'thumbnail': HOST + thumbnail_from_webpath(webpath),
                'original': HOST + webpath,
                # 'webformatURL': HOST+'data/images/'+'IMG20170819123559.jpg',
                'tags': get_tag(webpath),
                'name':personname,
                'details': get_img_detail(relpath,cursor),
                'avatar':HOST+avatars[i]
            }
            group.append(img)
        imgs.append(group)
        personnames.append(personname)
        num += len(group)
    detect.close()
    return {'names':personnames,'imgs':imgs}

@detectapp.route('/blur_detect/<path:dir>',methods=['GET'])
def blur_detect(dir):
    thres = settings['blur']
    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()
    filt = True
    if(dir=='__all__'):
        filt = False
    blur_imgs,num = [],0
    if(not CachedBlurImg==[]):
        for webpath,ft in CachedBlurImg:
            if filt and not webpath_belongto_dir(webpath,dir): continue#按文件夹过滤
            relpath = relpath_from_webpath(webpath)
            if not relpath or ft > thres: continue
            img = {
                 'id': relpath,
                 'index': num,
                'webpath': webpath,
                 'thumbnail': HOST + thumbnail_from_webpath(webpath),
                 'original':HOST + webpath,
                'details': get_img_detail(relpath,cursor),
                 # 'webformatURL': HOST+'data/images/'+'IMG20170819123559.jpg',
                 'tags': get_tag(webpath),
                 'ft':ft
            }
            num+=1
            blur_imgs.append(img)
    return {'imgs':blur_imgs,'total':num}

@detectapp.route('/screenshot/<path:dir>',methods=['GET'])
def screenshot(dir):
    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()
    dirs = [dir]
    if (dir == '__all__'):
        dirs = PathDict
    imgs,num = [],0
    for webdir in dirs:
        reldir = PathDict[webdir]
        relpaths = get_img_paths(reldir)
        for relpath in relpaths:
            file = relpath.rsplit('/',1)[1]
            if(is_screen_shot(relpath)):
                img = {
                    'id': relpath,
                    'index': num,
                    'webpath': webdir+'/'+file,
                    'thumbnail': HOST + '/'+webdir+'/.thumbnail/'+file,
                    'original': HOST + '/'+webdir+'/'+file,
                    'tags': get_tag(webdir+'/'+file),
                    'details': get_img_detail(relpath,cursor),
                }
                imgs.append(img)
                num += 1

    detect.close()
    return {'imgs':imgs,'total':num}

@detectapp.route('/fat/<path:dir>',methods=['GET'])
def fat(dir):
    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()
    dirs = [dir]
    if (dir == '__all__'):
        dirs = PathDict
    imgs, num = [], 0
    for webdir in dirs:
        reldir = PathDict[webdir]
        for root, dirs, files in os.walk(str(reldir)):
            dirs[:] = []

            for file in files:
                if not is_allowed_ext(file): continue

                details = get_img_detail(root+'/'+file,cursor)
                size_group = 0
                if details[0]<102400: size_group=0
                elif details[0]<1024*1024: size_group=1
                else: size_group = math.ceil(details[0]/(1024*1024))
                if(size_group>7):size_group=7
                img = {
                    'id': root+'/'+file,
                    'index': num,
                    'webpath': webdir + '/' + file,
                    'thumbnail': HOST + '/' + webdir + '/.thumbnail/' + file,
                    'original': HOST + '/' + webdir + '/' + file,
                    'tags': get_tag(webdir + '/' + file),
                    'details': details,
                    'size_group':size_group
                }
                imgs.append(img)
                num += 1
    imgs.sort(key=lambda x: x['details'][0], reverse=True)
    for i,img in enumerate(imgs):#修正index
        img['index'] = i
    detect.close()
    return {'imgs': imgs,'total':num}

@detectapp.route('/bLur_screen_fat/__all__',methods=['GET'])
def bLur_screen_fat_all():
    blurs, screens, sizes = [], [], []
    fats,thicks = [],[]
    for dir in PathDict:
        blurs += blur_detect(dir)['imgs']
        screens += screenshot(dir)['imgs']
        sizes += fat(dir)['imgs']
    for img in sizes:
        if(img['size_group']==0) : thicks.append(img)
        elif(img['size_group']>1): fats.append(img)
    return {'blurs':blurs[:10],'screenshots':screens,'fats':fats[:10],'thicks':thicks}

@detectapp.route('/bLur_screen_fat/<path:dir>',methods=['GET'])
def bLur_screen_fat(dir):
    blurs,screens,sizes = [],[],[]
    fats, thicks = [], []
    blurs = blur_detect(dir)['imgs']
    screens = screenshot(dir)['imgs']
    sizes = fat(dir)['imgs']
    for img in sizes:
        if(img['size_group']==0) : thicks.append(img)
        elif(img['size_group']>1): fats.append(img)
    return {'blurs':blurs,'screenshots':screens,'fats':fats[:10],'thicks':thicks}

@detectapp.route('/class_clear',methods=['POST'])
def class_clear():
    thres = settings['blur']
    detect = sqlite3.connect(database_file_path)  # 连接数据库
    cursor = detect.cursor()
    paths = request.json['paths']
    grouplen = len(paths)
    blurs,num = [],0
    webpaths = [webpath_from_relpath(path) for path in paths]
    for i,img_fm in enumerate(CachedBlurImg):
        if(img_fm[0] in webpaths):
            webpath = img_fm[0]
            relpath = relpath_from_webpath(webpath)
            if not relpath or img_fm[1]>thres: continue
            img = {
                'id': relpath,
                'index': num,
                'thumbnail': HOST + thumbnail_from_webpath(webpath),
                'original': HOST + webpath,
                'details': get_img_detail(relpath, cursor),
                # 'webformatURL': HOST+'data/images/'+'IMG20170819123559.jpg',
                'tags': get_tag(webpath),
                'ft': img_fm[1]
            }
            blurs.append(img)
            num+=1
    blurs.sort(key=lambda x:x['ft'])
    fats,num = [],0
    for path in paths:
        webpath = webpath_from_relpath(path)
        img = {
            'id': path,
            'index': num,
            'webpath': webpath,
            'thumbnail': HOST + thumbnail_from_webpath(webpath),
            'original': HOST + webpath,
            'details': get_img_detail(path, cursor),
            # 'webformatURL': HOST+'data/images/'+'IMG20170819123559.jpg',
            'tags': get_tag(webpath),
        }
        fats.append(img)
        num+=1
    fats.sort(key=lambda x:x['details'][0],reverse=True)
    detect.close()
    return {'blurs':blurs,'fats':fats[:grouplen//3]}



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