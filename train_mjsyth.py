import tensorflow as tf
from deployment import model_deploy
# from net import model_save as model
import os
slim = tf.contrib.slim
import time
from net import model
from dataset import read_utils
from tensorflow.python import debug as tf_debug

batch_size = 32
num_readers = 4
num_epochs = 2
checkpoint_dir = './tmp/'

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'length': 'label length',
    'label': 'A list of labels, one per each object.',
}


# =========================================================================== #
# Main
# =========================================================================== #
def main(_):

    tf.logging.set_verbosity(tf.logging.DEBUG)#设置显示的log的阈值

    with tf.Graph().as_default():

        deploy_config = model_deploy.DeploymentConfig()
        # Create global_step.
        global_step = tf.Variable(0,name='global_step',trainable=False)

        file_name = os.path.join("./tfrecord", "mjtrain_690_999.tfrecords")

        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   100000, 0.96, staircase=True)
        tf.summary.scalar("learning_rate",learning_rate)
        sh_images, sh_labels, sh_length= read_utils.inputs( filename = file_name,batch_size=batch_size,
                                num_epochs=num_epochs)




        crnn = model.CRNNNet()
        logits, inputs, seq_len, W, b = crnn.net(sh_images)

        cost = crnn.losses(sh_labels,logits, seq_len)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(loss=cost,global_step=global_step)
        tf.summary.scalar("cost",cost)


        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)


        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), sh_labels))
        tf.summary.scalar("edit_distance",acc)

        sess = tf.Session()

        save = tf.train.Saver(max_to_keep=2)
        if tf.train.latest_checkpoint(checkpoint_dir) is None:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)            # Start input enqueue threads.
        else:
            save.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))
            sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter('./tmp/my-model', sess.graph)

        try:
            step = global_step
            while not coord.should_stop():
                start_time = time.time()

                _,merged_t,val_cost, val_ler, lr, step = sess.run([optimizer,merged,cost, acc, learning_rate, global_step])

                duration = time.time() - start_time

                print("cost", val_cost)
                file_writer.add_summary(merged_t,step)
                # Print an overview fairly often.
                if step % 10 == 0:
                    print('Step %d:  acc %.3f (%.3f sec)' % (step,val_ler,duration))
                    save.save(sess,"./tmp/crnn-model.ckpt",global_step = global_step)
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

