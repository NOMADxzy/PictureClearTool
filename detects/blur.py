import argparse,_thread,cv2,pickle,os,sqlite3
import json

import numpy as np
from imutils import paths

from tools.general import get_img_paths,PathDict,webpath_from_relpath,\
    settings,relpath_from_webpath,executor
from tools.val import database_file_path

from remotes.RPC import extract_feat


class ImageFeatures():
    def __init__(self,noinit=False):
        if noinit: return
        self.CachedBlurImg = []
        self.feats = []
        self.names=[]
        try:
            featurestable = sqlite3.connect(database_file_path)
        except:
            print('-----重复初始化-----')
            return 
        cursor = featurestable.cursor()
        cursor.execute('select * from blur')
        results = cursor.fetchall()
        if not results is None:
            thres = settings['blur']
            for webpath, fm,ft in results:
                if not relpath_from_webpath(webpath):  # 图片不存在，移除
                    cursor.execute('delete from blur where webpath = ?', (webpath,))
                    cursor.execute('delete from detail where webpath = ?', (webpath,))
                # elif fm < thres:  # 小于清晰度阈值，加入模糊列表
                    # CachedBlurImg.append((webpath,fm))
                self.names.append(webpath)
                self.feats.append(json.loads(ft))
        featurestable.commit()
        featurestable.close()

        f = executor.submit(self.compute_blur)
        self.CachedBlurImg = f.result()
    
    
    def preImgOps(self,relpath):
        """
        图像的预处理操作
        """
        img = cv2.imread(relpath)  # 读取图片
        # 预处理操作
        reImg = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)  #
        img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY)  # 将图片压缩为单通道的灰度图
        return img2gray, reImg
    
    
    def compute_laplace(self,relpath):
        image,reImg = self.preImgOps(relpath)
        return cv2.Laplacian(image, cv2.CV_64F).var()
    
    
    def compute_SMD2(self,relpath):
        """
        灰度方差乘积
        """
        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(relpath)
        f=np.matrix(img2gray)/255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i+1,j]-f[i,j])*np.abs(f[i,j]-f[i,j+1])
        return score
    
    
    def run_blur_detect(self,webdir, thres = settings['blur'], score_func='SMD'):
        """
        遍历文件夹中所有图片，小于阈值的加入到模糊图片列表
        """
        updated = 0
    
        featurestable = sqlite3.connect(database_file_path)
        cursor = featurestable.cursor()
    
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
                    fm = self.compute_SMD2(imagePath)
                else:
                    fm = self.compute_laplace(imagePath)

                ft = extract_feat(imagePath)
                self.names.append(webpath)
                self.feats.append(ft)
                print("(retrieval) extracting feature from image No. %d ," + imagePath + "; " )
    
                cursor.execute('insert into blur values (?,?,?)', (webpath, fm,json.dumps(ft)))
    
            else:
                webpath,fm,ft = result
            if fm < thres:
                updated += 1
                self.CachedBlurImg.append((webpath,fm))

        featurestable.commit()
        featurestable.close()
        return updated
    
    def compute_blur(self,dir=None):
        updated = 0
    
        print('blur detect thres = ' + str(settings['blur']))
        for webdir in PathDict:
            if(dir and not webdir == dir): continue
            updated += self.run_blur_detect(webdir)
    
        if updated>0:
            print('(blur) find new blur images '+str(updated))
            self.CachedBlurImg.sort(key=lambda x:x[1])
    
        print("blur service stand by")
        return self.CachedBlurImg

imageFeatures = ImageFeatures()
    

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