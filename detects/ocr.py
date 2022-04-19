import easyocr
from PIL import Image


reader = easyocr.Reader(['ch_sim','en'])
def read(relpath):
    result = reader.readtext(relpath)
    width = Image.open(relpath).size[1]
    thres_w = width//6

    def takeX(tup):  # 计算当前框的中心线高度
        arr = tup[0]
        return (arr[0][1] + arr[1][1] + arr[2][1] + arr[3][1]) / 4

    def takeH(tup):  # 计算当前框的y方向长度
        arr = tup[0]
        return (arr[2][1] + arr[3][1] - arr[0][1] - arr[1][1]) / 2

    reshaped,float_left = [[]],[]
    if(result[0][0][0][0])<50: float_left.append(True)
    else: float_left.append(False)
    lastx = 0
    row = 0
    for tup in result:
        x = takeX(tup)
        h = takeH(tup)
        if x - lastx < h * 0.75:  # 和本行起始位置的差值大于当前元素高度的四分之三时视为下一行
            reshaped[row].append(tup[1])
        else:  # 新的一行
            reshaped.append([tup[1]])
            if (tup[0][0][0] < thres_w):
                float_left.append(True)
            else:
                float_left.append(False)
            lastx = x
            row += 1
    return reshaped,float_left