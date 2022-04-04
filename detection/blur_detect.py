from imutils import paths
import argparse
import cv2
from tools.general import get_img_paths,PathDict

CachedBlurImg = []


def conpute_laplace(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def run_blur_detect(webdir):
    thres = 70
    relpath = PathDict[webdir]
    img_paths = get_img_paths(relpath)
    blur_paths = []

    # 遍历每一张图片
    for imagePath in img_paths:
        # 读取图片
        image = cv2.imread(imagePath)
        # 将图片转换为灰度图片
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 计算灰度图片的方差
        fm = conpute_laplace(gray)
        #小于阈值不模糊
        if fm < thres:
            blur_paths.append((webdir+'/'+imagePath.rsplit('/',1)[1],fm))

        # 显示结果
    blur_paths.sort(key=lambda x:x[1])
    return blur_paths#webpath格式

for webdir in PathDict:
    CachedBlurImg += run_blur_detect(webdir)

if __name__ == '__main__':
    # 设置参数
    path = 'pics/'
    img_paths = get_img_paths('test_blur')

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