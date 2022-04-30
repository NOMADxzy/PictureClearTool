from pathlib import Path
import sqlite3,os
#数据库相关
database_file_path = 'tools/detect_results.db'
if not os.path.exists(database_file_path):
    detects = sqlite3.connect(database_file_path)
    cursor = detects.cursor()
    cursor.execute('''CREATE TABLE detail
    (
    	relpath
    		primary key,
    	details BLOB not null
    );
    ''')
    cursor.execute('''CREATE TABLE TagTable
    (
    	path TEXT
    		primary key,
    	boxs BLOB not null,
    	tag BLOB not null
    , del int default 0);''')
    cursor.execute('''CREATE TABLE TagGroupTable(
        tag INTEGER PRIMARY KEY ,
        imgs BLOB not null
    );''')
    cursor.execute('''CREATE TABLE Settings
    (
    	key text,
    	value BLOB
    );''')
    detects.commit()
    detects.close()

#数据相关
pathdict_file_path = 'tools/PathDict.pkl'
face_zip_path = 'tools/faces_zip.pkl'
blur_zip_path = 'tools/blurs_zip.pkl'
retrieval_file_path = 'tools/feature.h5'
settings_file_path = 'tools/setting.cfg'
yolo_weights_paths = ['weights/yolov5x6.pt','weights/yolov5m.pt','weights/yolov5s.pt']

#训练相关
cls_idx_base = 80 #后训练模型的类别索引号起点
train_dic = str(Path.joinpath(Path.cwd(),'data/fromuser'))

#flask相关
PORT = 9000

if __name__== '__main__':
    print(Path.joinpath(Path.cwd(),'data/fromuser'))
