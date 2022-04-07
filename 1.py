from PIL import Image
from PIL.ExifTags import TAGS

img = Image.open('pics/Screenshot_2017-07-21-08-59-03-05.png')
info = img._getexif()
if info is None:
    print("No Info")
else:
    for k, v in info.items():
        nice = TAGS.get(k, k)
        print('%s (%s) = %s' % (nice, k, v))

