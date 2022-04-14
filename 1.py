from PIL import Image
from PIL.ExifTags import TAGS
from tools.general import get_img_detail,is_screen_shot

# img = Image.open('pics/IMG_20180603_002316.jpg')
relpath = 'pics/8502299.jpg'
# img = Image.open(relpath)
# info = img._getexif()
# if info is None:
#     print("No Info")
# else:
#     for k, v in info.items():
#         nice = TAGS.get(k, k)
#         print('%s (%s) = %s' % (nice, k, v))
import os,time

# print(get_img_detail(relpath))
# print(is_screen_shot(relpath))


