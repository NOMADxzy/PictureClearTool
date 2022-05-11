##master分支：<br/>
####mac 系统下运行

python = 3.7

安装pytorch
https://pytorch.org/get-started/locally/

pip install -r requirements.txt

python app.py

初次运行easyocr下载模型失败的话需手动下载english_g2,zh_sim_g2,craft_mlt_25k
解压到~/.EasyOCR/model下
https://www.jaided.ai/easyocr/modelhub/

##rpc分支<br/>
pyinstaller打包
调用rpc服务 https://gitee.com/xu_zuyun/piclear_prc_server.git
