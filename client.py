import requests
import os,json

# for i in os.listdir("data/temp"):
#     image = open("data/temp/"+i,'rb')
#     payload = {'file':image}
# r = requests.post(" http://localhost:5000/predict", files=payload).json()
# print(r)
# res = requests.post("http://localhost:5000/detect/",json={'source':'data/images/bus.jpg'}).json()
# print(res)

# res = requests.post("http://localhost:5000/delete",json={'paths':['./save.jpg']})
# print(res)

res = requests.post("http://localhost:5000/import_dir",json={'dir':'/Users/macos/Pictures/leetcode'})
print(res)




