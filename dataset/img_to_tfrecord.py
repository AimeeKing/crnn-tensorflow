import tensorflow as tf

import os
import sys
import glob
from dataset.utils import _get_output_filename,int64_feature,bytes_feature


def img_to_tfrecord(image_dir,text_dir,text_name,output_dir,name):
    """

    :param image_dir: image_dir just like "data/Challenge2_Training_Task12_Images/*.jpg"
    :param text_dir: label file dir
    :param text_name: label file name
    :param output_dir: output dir
    :param name: output file name
    :return: NULL
    """

    tf_filename = _get_output_filename(output_dir, name)
    imgDirs = []
    imgLists = glob.glob(image_dir)  # return a list
    labels = []
    indexs = []
    shapes = []
    bf = open(os.path.join(text_dir, text_name)).read().splitlines()
    for idx in bf:
        shape = []
        spt = idx.split(' ')
        indexs.append(spt[0])
        labels.append(spt[1])
        shape.append(spt[2])
        shape.append(spt[3])
        shapes.append(shape)
    for item in imgLists:
        imgDirs.append(item)
    image_format = b'JPEG'
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, filename in enumerate(imgDirs):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(imgLists)))
            sys.stdout.flush()
            image_data = tf.gfile.FastGFile(filename, 'rb').read()

            # with tf.Session() as sess:
            #     image = tf.image.decode_jpeg(image_data)
            #     image = sess.run(image)
            #     print(image.shape)#(32, 100, 3)


            example = tf.train.Example(features=tf.train.Features(feature={"label": bytes_feature(labels[i]),
                                                                           "index":int64_feature(indexs[i]),
                                                                           'image/shape': int64_feature(shapes[i]),
                                                                           "image/encoded": bytes_feature(image_data),
                                                                           'image/format': bytes_feature(image_format)}))
            tfrecord_writer.write(example.SerializeToString())
    print('\nFinished converting the dataset!')


image_dir = "data/ICDAR_CROP/*.jpg"
text_dir ="data/ICDAR_CROP"
text_name = "labels.txt"
output_dir = "data/tfrecord"
name = "train"
img_to_tfrecord(image_dir,text_dir,text_name,output_dir,name)