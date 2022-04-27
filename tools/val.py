from pathlib import Path

pathdict_file_path = 'tools/PathDict.pkl'
face_zip_path = 'tools/faces_zip.pkl'
blur_zip_path = 'tools/blurs_zip.pkl'
database_file_path = 'tools/detect_results.db'
retrieval_file_path = 'tools/feature.h5'
settings_file_path = 'tools/setting.cfg'
yolo_weights_paths = ['weights/yolov5x6.pt','weights/yolov5m.pt','weights/yolov5s.pt']

#训练相关
cls_idx_base = 90 #后训练模型的类别索引号起点
train_dic = str(Path.joinpath(Path.cwd(),'data/datasets/fromuser'))

if __name__== '__main__':
    print(Path.joinpath(Path.cwd(),'data/datasets/fromuser'))
