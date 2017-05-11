import Image
import glob
import math

def binarizing(img,threshold): #input: gray image
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img

def depoint(img):   #input: gray image
    pixdata = img.load()
    w,h = img.size
    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y-1] > 245:
                count = count + 1
            if pixdata[x,y+1] > 245:
                count = count + 1
            if pixdata[x-1,y] > 245:
                count = count + 1
            if pixdata[x+1,y] > 245:
                count = count + 1
            if count > 2:
                pixdata[x,y] = 255
    return img

img_dirs = glob.glob("./demo/*.jpg")
for i,img_dir in enumerate(img_dirs):
    print("index",i,"image_dir",img_dir)
index = input("index\n")
dir = img_dirs[int(index)]
img = Image.open(dir)
size = img.size
print(size)
img.show()
# img = img.resize([math.ceil(size[0]*(32/size[1])),32])
# print(img.size)
# img.show()
img = img.convert('L')
img.show()
img = depoint(img)
img.show()