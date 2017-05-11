import tensorflow as tf
from net import model
import Image
from dataset.utils import  load_label_from_img_dir,encode_label,sparse_tensor_to_str
import glob
import os
import math
checkpoint_dir = './tmp/'

import numpy as np

def load_image(img_dir):
    """
    :param img_dir:
    :return:img_data
     load image and resize it
    """
    img = Image.open(img_dir)
    size = img.size
    width = math.ceil(size[0] * (32 / size[1]))
    img = img.resize([width, 32])
    im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((img.size[1], img.size[0], 3))
    im_arr = im_arr.astype(np.float32) * (1. / 255) - 0.5

    return im_arr,width


def prepare_data(img_dir):
    """
    :param img_dir:
    :return:
    """
    # first load image and label
    image_raw,width = load_image(img_dir)
    label = load_label_from_img_dir(img_dir)
    label = label.lower()
    return image_raw,label,width


width_input = tf.placeholder(tf.int32, shape=())
img_input = tf.placeholder(tf.float32, shape=(None, None, 3))
tf.reshape(img_input,shape=(32,-1,3))
img_4d = tf.expand_dims(img_input, 0)


# define the crnn net
crnn_params = model.CRNNNet.default_params._replace(batch_size=1)  # ,seq_length=int(width/4+1)
crnn = model.CRNNNet(crnn_params)
logits, inputs, seq_len, W, b = crnn.net(img_4d, width=width_input)

decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

saver = tf.train.Saver()

sess = tf.Session()
dir = tf.train.latest_checkpoint(checkpoint_dir)
saver.restore(sess, dir)
sess.run(tf.local_variables_initializer())
print("Model restore!")


def recognize_img(img_dir):
    img_raw,label,width = prepare_data(img_dir)

    decoded_s = sess.run([decoded],feed_dict={img_input:img_raw,width_input:width})
    # print(decoded_s[0])
    str = sparse_tensor_to_str(decoded_s[0])
    print("label",label)
    print('识别结果',str)



def main(_):
    img_dirs = glob.glob(os.path.join("demo/","*.jpg"))
    for i,img_dir in enumerate(img_dirs):
        print("index：",i,"name",img_dir)
    #index = int(input("the index choose is :"))
        recognize_img(img_dirs[i])



if __name__ =="__main__":
    tf.app.run()






