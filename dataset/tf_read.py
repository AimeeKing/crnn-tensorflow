""""
这个没啥用的，我用来打草稿的
"""

import tensorflow as tf
from deployment import model_deploy
import os
from tensorflow.python.ops import array_ops
slim = tf.contrib.slim
import time
from net import model
import numpy as np
from tensorflow.python import debug as tf_debug

batch_size = 32
num_readers = 4
num_epochs = 2

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'index': 'index of the image',
    'label': 'A list of labels, one per each object.',
}
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.
    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image
def _ImageDimensions(image):
    """Returns the dimensions of an image tensor.
    Args:
      image: A 3-D Tensor of shape `[height, width, channels]`.
    Returns:
      A list of `[height, width, channels]` corresponding to the dimensions of the
        input image.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]
def resize_image(image ,size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    """Resize an image and bounding boxes.
    """
    # Resize image.
    with tf.name_scope('resize_image'):
        height_i, width_i, channels = _ImageDimensions(image)

        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        channels = 3
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image



def preprocess_for_train(image,label ,scope='crnn_preprocessing_train'):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

    Args:
        image: A `Tensor` representing an image of arbitrary size.
    Returns:
        A preprocessed image.
    """
    fast_mode = False
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)# convert image as a tf.float32 tensor
            image_s = tf.expand_dims(image, 0)
            tf.summary.image("image",image_s)

        image =  tf.image.rgb_to_grayscale(image)
        tf.summary.image("gray",image)
        return  image,label,


def get_split(file_dir,reader = None):
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),#三个参数：shape,type,default_value
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'label': tf.FixedLenFeature((), tf.string, default_value='unknow'),
        'index': tf.FixedLenFeature([1], tf.int64),
    }
    items_to_handlers = {
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'index': slim.tfexample_decoder.Tensor('index'),
        'label': slim.tfexample_decoder.Tensor('label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    return slim.dataset.Dataset(
            data_sources=file_dir,
            reader=reader,
            decoder=decoder,
            num_samples=849,
            items_to_descriptions = ITEMS_TO_DESCRIPTIONS
            )


# =========================================================================== #
# Main
# =========================================================================== #
def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)#设置显示的log的阈值


    with tf.Graph().as_default():
        # Config model_deploy. Keep TF Slim Models structure.
        # Useful if want to need multiple GPUs and/or servers in the future.
        deploy_config = model_deploy.DeploymentConfig()
        # Create global_step.
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()
        file_name = os.path.join("../tfrecord", "train.tfrecords")
        def read_and_decode(filename,num_epochs):  # read iris_contact.tfrecords
            filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=num_epochs)
            reader = tf.TFRecordReader()
            print(filename_queue)
            _, serialized_example = reader.read(filename_queue)  # return file_name and file
            features = tf.parse_single_example(serialized_example,
                                               features={
                                                   'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                                                   # 三个参数：shape,type,default_value
                                                   'image/format': tf.FixedLenFeature((), tf.string,
                                                                                      default_value='jpeg'),
                                                   'image/shape': tf.FixedLenFeature([2], tf.int64),
                                                   'label': tf.FixedLenFeature((), tf.string, default_value='unknow'),
                                                   'index': tf.FixedLenFeature([1], tf.int64)
                                               })  # return image and label

            # img = tf.decode_raw(features['image/encoded'], tf.uint8)
            img = tf.image.decode_jpeg(features['image/encoded'])
            shape = features["image/shape"]
            img = tf.reshape(img, [32, 100, 3]) #  reshape image to 512*80*3
            img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # throw img tensor
            label = features['label'] # throw label tensor
            index = features["index"]
            return img, label,shape,index

        def preprocess(image_raw):
            image = tf.image.decode_jpeg(tf.image.encode_jpeg(image_raw))
            return resize_image(image,(100,32))


        def inputs( batch_size, num_epochs,filename):
            """Reads input data num_epochs times.
            Args:
              train: Selects between the training (True) and validation (False) data.
              batch_size: Number of examples per returned batch.
              num_epochs: Number of times to read the input data, or 0/None to
                 train forever.
            Returns:
              A tuple (images, labels), where:
              * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
                in the range [-0.5, 0.5].
              * labels is an int32 tensor with shape [batch_size] with the true label,
                a number in the range [0, mnist.NUM_CLASSES).
              Note that an tf.train.QueueRunner is added to the graph, which
              must be run using e.g. tf.train.start_queue_runners().
            """
            if not num_epochs: num_epochs = None
            #filename = os.path.join(file_dir)

            with tf.name_scope('input'):
                # Even when reading in multiple threads, share the filename
                # queue.
                image, label,shape,index= read_and_decode(filename,num_epochs)

                #image = preprocess(image)
                # Shuffle the examples and collect them into batch_size batches.
                # (Internally uses a RandomShuffleQueue.)
                # We run this in two threads to avoid being a bottleneck.
                images, shuffle_labels,sshape,sindex = tf.train.shuffle_batch(
                    [image, label,shape,index], batch_size=batch_size, num_threads=2,
                    capacity=1000 + 3 * batch_size,
                    # Ensures a minimum amount of shuffling of examples.
                    min_after_dequeue=100)

                return images, shuffle_labels,sshape,sindex


        with tf.Graph().as_default():
            # Input images and labels.
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       100000, 0.96, staircase=True)
            images, shuffle_labels, sshape, sindex = inputs( filename = file_name,batch_size=batch_size,
                                    num_epochs=num_epochs)

            crnn = model.CRNNNet()
            logits, inputs, seq_len, W, b = crnn.net(images)

            shuffle_labels = ['123456','123','12342']
            labels = shuffle_labels

            def sparse_tuple_from(sequences, dtype=np.int32):
                """Create a sparse representention of x.
                Args:
                    sequences: a list of lists of type dtype where each element is a sequence
                Returns:
                    A tuple with (indices, values, shape)
                """
                indices = []
                values = []

                for n, seq in enumerate(sequences):
                    indices.extend(zip([n] * len(seq), range(len(seq))))
                    values.extend(seq)

                indices = np.asarray(indices, dtype=np.int64)
                values = np.asarray(values, dtype=dtype)
                shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

                return indices, values, shape

            sparse_labels = sparse_tuple_from(labels)

            cost = crnn.losses(sparse_labels,logits, seq_len)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(loss=cost,global_step=global_step)

            # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
            # (it's slower but you'll get better results)
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

            # Accuracy: label error rate
            acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sparse_labels))

            sess = tf.Session()
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            try:
                step = 0
                while not coord.should_stop():
                    start_time = time.time()

                    # Run one step of the model.  The return values are
                    # the activations from the `train_op` (which is
                    # discarded) and the `loss` op.  To inspect the values
                    # of your ops or variables, you may include them in
                    # the list passed to sess.run() and the value tensors
                    # will be returned in the tuple from the call.
                    #timages, tsparse_labels, tsshape, tsindex = sess.run([images, sparse_labels, sshape, sindex])

                    val_cost, val_ler, lr, step = sess.run([cost, acc, learning_rate, global_step])

                    duration = time.time() - start_time

                    print(val_cost)

                    # Print an overview fairly often.
                    if step % 10 == 0:
                        print('Step %d:  (%.3f sec)' % (step,duration))
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (num_epochs, step))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

                # Wait for threads to finish.
            coord.join(threads)
            sess.close()
if __name__ == '__main__':
    tf.app.run()

