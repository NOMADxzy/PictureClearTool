import argparse,_thread,cv2,pickle,os,sqlite3
import numpy as np
from imutils import paths

from tools.general import get_img_paths,PathDict,webpath_from_relpath,\
    settings,relpath_from_webpath
from tools.val import database_file_path

CachedBlurImg = []


def init():
    blur_res = sqlite3.connect(database_file_path)
    cursor = blur_res.cursor()
    cursor.execute('select * from blur')
    results = cursor.fetchall()
    if not results is None:
        thres = settings['blur']
        for webpath, fm in results:
            if not relpath_from_webpath(webpath):  # 图片不存在，移除
                cursor.execute('delete from blur where webpath = ?', (webpath,))
            # elif fm < thres:  # 小于清晰度阈值，加入模糊列表
                # CachedBlurImg.append((webpath,fm))
    blur_res.commit()
    blur_res.close()


def preImgOps(relpath):
    """
    图像的预处理操作
    """
    img = cv2.imread(relpath)  # 读取图片
    # 预处理操作
    reImg = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)  #
    img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY)  # 将图片压缩为单通道的灰度图
    return img2gray, reImg


def compute_laplace(relpath):
    image,reImg = preImgOps(relpath)
    return cv2.Laplacian(image, cv2.CV_64F).var()


def compute_SMD2(relpath):
    """
    灰度方差乘积
    """
    # step 1 图像的预处理
    img2gray, reImg = preImgOps(relpath)
    f=np.matrix(img2gray)/255.0
    x, y = f.shape
    score = 0
    for i in range(x - 1):
        for j in range(y - 1):
            score += np.abs(f[i+1,j]-f[i,j])*np.abs(f[i,j]-f[i,j+1])
    return score


def run_blur_detect(webdir, thres = settings['blur'], score_func='SMD2'):
    """
    遍历文件夹中所有图片，小于阈值的加入到模糊图片列表
    """
    updated = 0

    blur_res = sqlite3.connect(database_file_path)
    cursor = blur_res.cursor()

    reldir = PathDict[webdir]
    if not os.path.isdir(reldir):
        print('(blur) ' + webdir + ' 文件夹不存在')
        return 0

    img_paths = get_img_paths(reldir,webpath=webdir)

    # 遍历每一张图片
    for webpath in img_paths:
        cursor.execute('select * from blur where webpath = ?',(webpath,))
        result = cursor.fetchone()

        if result is None:
            # 读取图片
            imagePath = relpath_from_webpath(webpath)
            print('(blur) compute image ' + imagePath + ' func: ' + score_func)

            # 计算灰度图片的方差
            if score_func == 'SMD2':
                fm = compute_SMD2(imagePath)
            else:
                fm = compute_laplace(imagePath)

            cursor.execute('insert into blur values (?,?)', (webpath, fm))
            print('(blur) new img ' + webpath + ' fm = ' + str(fm))
        else:
            webpath,fm = result
        if fm < thres:
            updated += 1
            CachedBlurImg.append((webpath,fm))
    blur_res.commit()
    blur_res.close()
    return updated

def compute_blur(dir=None):
    updated = 0

    print('blur detect thres = ' + str(settings['blur']))
    for webdir in PathDict:
        if(dir and not webdir == dir): continue
        updated += run_blur_detect(webdir)

    if updated>0:
        print('(blur) find new blur images '+str(updated))
        CachedBlurImg.sort(key=lambda x:x[1])

    print("blur service stand by")

def run():
    init()
    compute_blur()

try:
    _thread.start_new_thread(run,())
except:
    print("模糊图片检测线程启动失败")

if __name__ == '__main__':
    # 设置参数
    path = '../test_blur'
    img_paths = get_img_paths(paths)

    # 遍历每一张图片
    for imagePath in paths.list_images(path):
        # 读取图片
        image = cv2.imread(imagePath)
        # 将图片转换为灰度图片
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 计算灰度图片的方差
        fm = compute_laplace(gray)
        text = "Not Blurry"

        # 设置输出的文字
        if fm < 100:
            text = "Blurry"

        # 显示结果
        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)