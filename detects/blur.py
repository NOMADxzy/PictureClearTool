from imutils import paths
import argparse,_thread
import cv2,pickle,os
from tools.general import get_img_paths,PathDict,webpath_from_relpath,settings
from tools.val import blur_zip_path

relpaths,fms,CachedBlurImg = [],[],[]
if(os.path.exists(blur_zip_path)):#存在则读取上次的结果
    if os.path.getsize(blur_zip_path)>100:
        with open(blur_zip_path,'rb') as file:
            try:
                blur_zip = pickle.load(file)
                relpaths,fms,CachedBlurImg = blur_zip
            except:
                print('文件错误')
            finally: file.close()

def purify():
    relpaths1,fms1,CachedBlurImg1 = [],[],[]
    for i,relpath in enumerate(relpaths):
        webpath = webpath_from_relpath(relpath)
        if (not os.path.exists(relpath) or not webpath):
            print('(blur) cant find :'+relpath+'do removing it')
        else:
            relpaths1.append(relpath)
            fms1.append(fms[i])
            CachedBlurImg1.append((webpath,fms1[i]))
    relpaths[:] = relpaths1
    fms[:] = fms1
    CachedBlurImg[:] = CachedBlurImg1
    with open(blur_zip_path, 'wb') as file:
        faces_zip = [relpaths, fms, CachedBlurImg]
        pickle.dump(faces_zip, file)
        file.close()

def conpute_laplace(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def run_blur_detect(webdir ,thres = settings['blur']):
    updated = 0
    reldir = PathDict[webdir]
    if(not os.path.isdir(reldir)):
        print('(blur) ' + webdir + ' 文件夹不存在')
        return 0
    img_paths = get_img_paths(reldir)
    # blur_paths = []

    # 遍历每一张图片
    for imagePath in img_paths:
        if(imagePath in relpaths): continue
        # 读取图片
        print('(blur) compute image fm: '+imagePath)
        image = cv2.imread(imagePath)
        # 将图片转换为灰度图片
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 计算灰度图片的方差
        fm = conpute_laplace(gray)
        relpaths.append(imagePath)
        fms.append(fm)
        #小于阈值模糊
        if fm < thres:
            print('(blur) new blur img '+imagePath + ' thres = '+str(fm))
            updated += 1
            CachedBlurImg.append((webdir+'/'+imagePath.rsplit('/',1)[1],fm))
        # 显示结果
    # blur_paths.sort(key=lambda x:x[1])
    # return blur_paths#webpath格式
    return updated

def compute_blur():
    updated = 0
    print('blur detect thres = ' + str(settings['blur']))
    for webdir in PathDict:
        updated += run_blur_detect(webdir)
    if(updated>0) :
        print('(blur) find new blur images '+str(updated))
        CachedBlurImg.sort(key=lambda x:x[1])
    with open(blur_zip_path, 'wb') as file:
        faces_zip = [relpaths, fms, CachedBlurImg]
        pickle.dump(faces_zip, file)
        file.close()
    print("blur service stand by")

def run():
    purify()
    compute_blur()

try:
    _thread.start_new_thread(run,())
except:
    print("计算模糊图片线程启动失败")

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
        fm = conpute_laplace(gray)
        text = "Not Blurry"

        # 设置输出的文字
        if fm < 100:
            text = "Blurry"

        # 显示结果
        cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        key = cv2.waitKey(0)