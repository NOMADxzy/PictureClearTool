# from PIL import Image
# from PIL.ExifTags import TAGS
#
# img = Image.open('pics/IMG20170611093917.jpg')
# info = img._getexif()
# if info is None:
#     print("No Info")
# else:
#     for k, v in info.items():
#         nice = TAGS.get(k, k)
#         print('%s (%s) = %s' % (nice, k, v))


import requests,sqlite3
def test_del():
    for root, dirs, files in os.walk('test_pics'):
        paths = []
        for file in files:
            path = root + '/' + file
            paths.append(path)
        res = requests.post("http://localhost:5000/delete", json={'paths': paths})
        print(res)
