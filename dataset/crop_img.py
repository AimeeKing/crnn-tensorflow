import os

import glob
import Image, ImageDraw

height = 32

# ground truth directory
gt_text_dir = "data/Challenge2_Training_Task1_GT"

# original images directory
image_dir = "data/Challenge2_Training_Task12_Images/*.jpg"
imgDirs = []
imgLists = glob.glob(image_dir)#return a list
print(imgLists)
# where to save the images with ground truth boxes
imgs_save_dir = "data/ICDAR_CROP"
text_save_dir = "labels.txt"

labels = []
shapes = []
count = 0
if not os.path.exists(imgs_save_dir):
    os.makedirs(imgs_save_dir)

for item in imgLists:
    imgDirs.append(item)

for img_dir in imgDirs:
    img = Image.open(img_dir)
    dr = ImageDraw.Draw(img)
    print(img.size)
    img_basename = os.path.basename(img_dir)
    (img_name, temp2) = os.path.splitext(img_basename)
    # open the ground truth text file
    img_gt_text_name = "gt_" + img_name + ".txt"
    print(img_gt_text_name)

    bf = open(os.path.join(gt_text_dir, img_gt_text_name)).read().splitlines()
    print(bf)
    for idx in bf:
        rect = []
        spt = idx.split(' ')
        print(spt)
        rect.append(float(spt[0]))
        rect.append(float(spt[1]))
        rect.append(float(spt[2]))
        rect.append(float(spt[3]))
        labels.append(spt[4])
        img_rect = img.crop((rect[0],rect[1],rect[2],rect[3]))
        print(img_rect.size)
        img_rect = img_rect.resize(tuple([100,32]))
        image_name = str("%06d"%(count+1))+temp2
        img_rect.save(os.path.join(imgs_save_dir, image_name))
        shapes.append(img_rect.size)
        count += 1
    #img.save(os.path.join(imgs_save_dir, img_basename))
txt_dir = os.path.join(imgs_save_dir, text_save_dir)
text_file = open(txt_dir,'w')
for i,label in enumerate(labels):
    text_file.write('%06d %s %f %f \n'%(i+1,label,shapes[i][0],shapes[i][1]))
text_file.close()