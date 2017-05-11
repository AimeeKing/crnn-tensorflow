"""
this is an implement to load img from mjsyth.tar
"""
import tensorflow as tf
import os
import glob
import sys
from dataset.utils import int64_feature,bytes_feature,load_label_from_imglist,load_image,encode_labels

tf_filename = os.path.join("../tfrecord","mjtrain_690_999.tfrecords")
imgDir = "../data/mnt/ramdisk/max/696_999/*"#训练时用/*


imgLists = []

Dirs = glob.glob(imgDir)
imgDirs = []
for imgDir in Dirs:
    if os.path.isdir(imgDir):
        imgDirs.extend(glob.glob(os.path.join(imgDir,"*")))

for imgDir in imgDirs:
    path = os.path.join(imgDir,"*.jpg")
    imgList = glob.glob(os.path.join(imgDir,"*.jpg"))
    imgLists.extend(imgList)

print("一共有%d个文件"%(len(imgLists)))

labels = load_label_from_imglist(imgLists)
labels_encord,lengths = encode_labels(labels)



with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
    for i, filename in enumerate(imgLists):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(imgLists)))
        sys.stdout.flush()
        try:
            image_data = load_image(filename)
            example = tf.train.Example(features=tf.train.Features(feature={"label/value": int64_feature(labels_encord[i]),
                                                                           "image/encoded": bytes_feature(image_data),
                                                                           "label/length":int64_feature(lengths[i]),
                                                                           'image/format': bytes_feature("jpeg")}))
            tfrecord_writer.write(example.SerializeToString())
        except Exception as e:
            print("Error: ",e)

print('\nFinished converting the dataset!')
