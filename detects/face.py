# -*- coding: UTF-8 -*-
import os, pickle, sqlite3
from PIL import Image
import numpy as np
from datetime import datetime

from tools.general import get_img_paths, PathDict, TagGroup, \
    relpath_from_webpath, executor,settings
from tools.val import face_zip_path,database_file_path

from remotes.RPC import Face_recognition
face_recognition = Face_recognition()


known_face_names, known_face_imgs, known_face_encodings, mislead = [], [], [], []  # 已提取的人脸数据
cached, avatars = [], []


if (os.path.exists(face_zip_path)):  # 存在则读取上次的结果
    with open(face_zip_path, 'rb') as file:
        try:
            if os.path.getsize(face_zip_path) > 100:
                face_zip = pickle.load(file)
                known_face_names, known_face_encodings, known_face_imgs, mislead = face_zip
        except:
            print('(face) 文件错误')
        finally:
            file.close()


def remove_duplicate(img_poses):
    webpaths = []
    img_poses1 = []
    for i, img_pos in enumerate(img_poses):
        if not img_pos[0] in webpaths:  # 确保某一人物下的所有图片不重复
            webpaths.append(img_pos[0])
            img_poses1.append(img_pos)
    return img_poses1


def purify():
    known_face_names1, known_face_imgs1, known_face_encodings1 = [], [], []
    for i, name in enumerate(known_face_names):
        imgs = remove_duplicate(known_face_imgs[i])  # 去重
        imgs1 = []  # 存放去除不存在的照片后的人脸图片

        for j, face in enumerate(imgs):
            if (relpath_from_webpath(face[0]) and os.path.exists(relpath_from_webpath(face[0]))):
                imgs1.append(face)
                cached.append(face[0])  # 已检测过的图片
        if len(imgs1) != 0:  # 该人物未清空
            known_face_names1.append(name)
            known_face_imgs1.append(imgs1)
            known_face_encodings1.append(list(known_face_encodings[i]))

    for webpath in mislead:  # mislead中去除不存在的照片
        if not relpath_from_webpath(webpath):
            mislead.remove(webpath)

    if len(known_face_names1)>0:
        with open(face_zip_path, 'wb') as file:  # 所有的更改写回文件
            faces_zip = [known_face_names1, known_face_encodings1, known_face_imgs1,mislead]
            pickle.dump(faces_zip, file)
            file.close()

    return known_face_names1, known_face_imgs1, known_face_encodings1


# 获得所有带person标签的文件
def get_paths():
    detect = sqlite3.connect(database_file_path)
    cursor = detect.cursor()
    cursor.execute("""select * from TagGroupTable where tag = ?""", (0,))
    t, imgs_dump = cursor.fetchone()
    webpaths = pickle.loads(imgs_dump)
    detect.close()
    return webpaths


def find(webpaths):
    print('face detect tolerant = ' + str(settings['face']))
    # 测试起始时间
    t1 = datetime.now()
    t10 = t1 - t1
    t20 = t10
    t30 = t10
    t40 = t10
    t50 = t10
    count_checked, count_copied = 0, 0
    print('cached: '+str(len(cached))+'mislead: '+str(len(mislead)))
    for webpath in webpaths:
        if (webpath in cached or webpath in mislead or not relpath_from_webpath(webpath)): continue  # 已检测过的照片/不含人脸的照片
        print('(face) check new img '+webpath)
        image_path = relpath_from_webpath(webpath)
        # 加载图片
        t00 = datetime.now()
        # unknown_image = face_recognition.load_image_file(image_path)
        t10 += datetime.now() - t00
        count_checked += 1

        # 找到图中所有人脸的位置
        t00 = datetime.now()
        face_locations = face_recognition.face_locations(image_path)
        if (len(face_locations) == 0): mislead.append(webpath)


        # face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=0, model="cnn")
        t20 += datetime.now() - t00

        # 根据位置加载人脸编码的列表
        t00 = datetime.now()
        face_encodings = face_recognition.face_encodings(image_path)
        t30 += datetime.now() - t00

        # 遍历所有人脸编码，与已知人脸对比
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 获得对比结果，tolerance值越低比对越严格
            if (len(known_face_encodings) == 0):  # 还没有任何人脸数据
                name = '未命名'
                known_face_names.append(name)
                known_face_encodings.append(face_encoding)
                known_face_imgs.append([(webpath, [left, top, right, bottom])])
                continue
            t00 = datetime.now()
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=settings['face'])
            t40 += datetime.now() - t00

            # 匹配到人脸
            if True in matches:
                ids = np.argwhere(matches)
                for id in ids:
                    id = id[0]
                    name = known_face_names[id]
                    print("(face) matched " + image_path + " : " + name)
                    known_face_imgs[id].append([webpath, [left, top, right, bottom]])
            else:  # 不和当前已有人脸匹配
                name = '未命名'
                print('(face) find new person : ' + webpath)
                known_face_names.append(name)
                known_face_encodings.append(face_encoding)
                known_face_imgs.append([[webpath, [left, top, right, bottom]]])

    with open(face_zip_path, 'wb') as file:
        faces_zip = [known_face_names, known_face_encodings, known_face_imgs, mislead]
        pickle.dump(faces_zip, file)
        file.close()

        # 测试结束时间
        t2 = datetime.now()
        # 显示总的时间开销
        print('%d pictures checked, and %d pictures copied with known faces.' % (count_checked, count_copied))
        print('time spend: %d seconds, %d microseconds.' % ((t2 - t1).seconds, (t2 - t1).microseconds))
        # print('load_image_file time: %d seconds, %d microseconds.' % (t10.seconds, t10.microseconds))
        # print('face_locations  time: %d seconds, %d microseconds.' % (t20.seconds, t20.microseconds))
        # print('face_encodings  time: %d seconds, %d microseconds.' % (t30.seconds, t30.microseconds))
        # print('compare_faces   time: %d seconds, %d microseconds.' % (t40.seconds, t40.microseconds))
        # print('shutil.copy     time: %d seconds, %d microseconds.' % (t50.seconds, t50.microseconds))

def make_avatar(webpath,pos):
    import uuid
    path_pre = 'temp/avatar/'
    relpath = relpath_from_webpath(webpath)
    if not relpath:
        print('make_avatar fail '+webpath)
        return None
    avatar = Image.open(relpath)
    avatar = avatar.crop(pos)
    avatar_path = path_pre+str(uuid.uuid1())+'.png'
    avatar.save(avatar_path)
    return avatar_path



def generate_avatar():
    paths = get_img_paths('temp/avatar/')
    for p in paths:
        os.remove(p)#删除之前的图片
    for i, group in enumerate(known_face_imgs):
        webpath, pos = group[0]
        relpath = relpath_from_webpath(webpath)
        if (not relpath):
            continue

        avatar = Image.open(relpath)
        avatar = avatar.crop(pos)
        try:
            name = '未命名' + str(i) if known_face_names[i] == '未命名' else known_face_names[i]
            avatar_path = 'temp/avatar/' + name + '.png'
            avatar.save(avatar_path)
            avatars.append(avatar_path)
        except:
            print('(face)保存' + webpath + '头像失败')
    print("(face) generated person avatar")


known_face_names, known_face_imgs, known_face_encodings = purify()  # 删除不存在的路径


def run():
    paths = get_paths()
    find(paths)  # 匹配所有新增的人脸照片
    print('find person num '+ str(known_face_names))
    # generate_avatar()
    print('face process stand by')

executor.submit(run)