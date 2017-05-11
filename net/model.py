import tensorflow as tf
import numpy as np
import math
from net import custom_layers
from collections import namedtuple
from tensorflow.contrib import rnn

slim = tf.contrib.slim

# =========================================================================== #
# RCNN class definition.
# =========================================================================== #
RCNNParams = namedtuple('RCNNParameters', ['ks',#kernel_size
                                         'ps',#padding_size
                                         'ss',#stride_size
                                         'nm',#In/Out size
                                         'imgH',
                                         'nc',
                                         'nclass',
                                         'nh',
                                         'n_rnn',
                                         'leakyRelu',
                                          'batch_size',
                                        'seq_length',
                                        'input_size',
                                        "reuse"
                                         ])


class CRNNNet(object):
    """

    """
    default_params = RCNNParams(
        ks=[3, 3, 3, 3, 3, 3, 2],  # kernel_size
        ps = [1, 1, 1, 1, 1, 1, 0], # padding_size
        ss = [1, 1, 1, 1, 1, 1, 1],  # stride_size
        nm = [64, 128, 256, 256, 512, 512, 512],# In/Out size
        leakyRelu = False,
        n_rnn =2,
        nh = 100,#size of the lstm hidden state
        imgH = 64,#the height / width of the input image to network
        nc = 1,
        nclass = 37,#0~9,a~z,还有一个啥都不是
        batch_size= 32,
        seq_length = 26,
        input_size = 512,
        reuse = None
           )





    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, RCNNParams):
            self.params = params
        else:
            self.params = CRNNNet.default_params

    # ======================================================================= #
    def net(self, inputs,is_test = False,width = None):
        """rcnn  network definition.
        """
        def BLSTM(inputs,num_hidden,num_layers,seq_len,num_classes,reuse=None):
            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell
            # cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

            # Stacking rnn cells
            stack = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([(tf.contrib.rnn.core_rnn_cell.GRUCell(num_hidden)) for i in range(num_layers)],
                                                              state_is_tuple=True)

            # The second output is the last state and we will no use that
            outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

            shape = tf.shape(inputs)
            batch_s, max_timesteps = shape[0], shape[1]

            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, num_hidden])

            # Truncated normal with mean 0 and stdev=0.1
            # Tip: Try another initialization
            # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
            W = tf.Variable(tf.truncated_normal([num_hidden,
                                                 num_classes],
                                                stddev=0.1), name="W")
            # Zero initialization
            # Tip: Is tf.zeros_initializer the same?
            b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

            # Doing the affine projection
            logits = tf.matmul(outputs, W) + b

            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_s, -1, num_classes])

            # Time major
            logits = tf.transpose(logits, (1, 0, 2))
            return logits, inputs, seq_len, W, b

        def BidirectionalLSTM(input,nHidden,nOut,seq_len,scope = "BLSTM"):#bLSTM 测试一
            # Prepare data shape to match `bidirectional_rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

            # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            def fulconn_layer(input_data, output_dim, activation_func=None):
                input_dim = int(input_data.get_shape()[1])
                W = tf.Variable(tf.random_normal([input_dim, output_dim]))
                b = tf.Variable(tf.random_normal([output_dim]))
                if activation_func:
                    return activation_func(tf.matmul(input_data, W) + b)
                else:
                    return tf.matmul(input_data, W) + b

            with tf.name_scope(scope):
                with tf.variable_scope('forward'):
                    lstm_fw_cell = rnn.LSTMCell(nHidden, forget_bias=1.0, state_is_tuple=True)
                with tf.variable_scope('backward'):
                    lstm_bw_cell = rnn.LSTMCell(nHidden, forget_bias=1.0, state_is_tuple=True)
                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell,
                                                                  inputs=input, sequence_length=seq_len,
                                                                  dtype=tf.float32, scope=scope)

                outputs = tf.concat(outputs, 2)

                outputs = tf.reshape(outputs, [-1, nHidden * 2])

                outputs = fulconn_layer(outputs, nOut)
                return outputs

                #outputs = tf.reshape(outputs, [64, 28, nOut])#这里要改

        def conv2d(inputs,i,batchNormalization = False):
            nOut = self.params.nm[i]
            kernel_size = self.params.ks[i]
            stride = self.params.ss[i]
            if i != 6:
                net = slim.conv2d(inputs,nOut,kernel_size,stride = stride,padding="SAME",scope = 'conv_{0}'.format(i),activation_fn=None)
            else:
                net = slim.conv2d(inputs, nOut, kernel_size, stride=stride, padding="VALID", scope='conv_{0}'.format(i),
                                  activation_fn=None)

            if(batchNormalization):
                net = slim.batch_norm(net,scope='batchnorm{0}'.format(i))

            net = tf.nn.relu(net)
            return net

        if is_test:
            print("inputs:", inputs.shape)
            inputs = tf.reshape(inputs, shape=(32, -1, 3))
            inputs = tf.expand_dims(inputs, 0)

        with tf.variable_scope("RCNN_net"):
            net = conv2d(inputs,0)#input batch_size*32*100*3 #net batch_size *32*100*64
            net = slim.max_pool2d(net, [2, 2], scope='pool1')#net batch_size *16*50*64
            print("poll_0 ", net.shape)
            net = conv2d(net,1)#batch_size*16*50*128
            net = slim.max_pool2d(net, [2, 2], scope='pool2')#batch_size *8*25*128 后面简写b
            print("poll_1 ", net.shape)
            net = conv2d(net,2,True) #b *8*25*256

            net = conv2d(net,3) #b*8*25*256
            net = custom_layers.pad2d(net, pad=(0, 1))#b*8*27*256
            net = slim.max_pool2d(net,[2,2],stride =[2,1],padding = "VALID")#b*4*26*256
            print('pool_3', net.shape)
            #net = slim.max_pool2d(net,[2,2],stride =[2,1],padding ="SAME")
            net = conv2d(net,4,True)#b*4*26*512
            net = conv2d(net,5)#b*4*26*512
            net = custom_layers.pad2d(net, pad=(0, 1))#b*4*28*512
            net = slim.max_pool2d(net,[2,2],stride=[2,1],padding = "VALID")#b*2*27*512
            print("pool_5", net.shape)
            net = conv2d(net,6,True)#b*1*26*512
            print("conv6",net.shape)

            net = tf.squeeze(net,[1])#B*26*512
            print("squeeze: ", net.shape)
            # net = tf.transpose(net, perm=(2,0,1)) #在pytroch中 LSTM函数顺序为seq_len,batch,input_size
            # print("transpose: ", net.shape)
            if width is None:
                seq_length = self.params.seq_length
            else:
                seq_length = tf.cast(width/4+1,tf.int32)
            seq_len = np.ones(self.params.batch_size) * seq_length
            # net = BidirectionalLSTM(net,self.params.nh,self.params.nh,seq_len=seq_len,scope="BLSTM_1")#？*512*100
            # net =tf.reshape(net,[self.params.batch_size,self.params.seq_length,self.params.nh])
            # net = BidirectionalLSTM(net, self.params.nh, self.params.nclass, seq_len=seq_len, scope="BLSTM_2")  # ？*512*100
            # net = tf.reshape(net, [self.params.batch_size, self.params.seq_length, self.params.nclass])
            result = BLSTM(net,self.params.nh,2,seq_len,self.params.nclass)
            return result


    def losses(self, targets,logits, seq_len,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        with tf.name_scope(scope):
            loss = tf.nn.ctc_loss(targets, logits, seq_len)
            cost = tf.reduce_mean(loss)

        return cost
