#!/usr/bin/env python
# -*- coding: UTF-8 -*-


#**************The code is adapted from the following files *********************************************************
#                     https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/cifar10-preact18-mixup.py
#                     and 
#                     https://github.com/tensorpack/tensorpack/blob/master/examples/basics/mnist-convnet.py
#******************************************************************************************************************** 

import numpy as np
import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import (
    varreplace, summary, get_current_tower_context, optimizer, gradproc)
from tensorpack.utils.fs import download, get_dataset_path
import gzip
import numpy
from six.moves import range
import scipy.io
from tensorpack.tfutils.summary import *

isAdaMixup = True # if False, run the benchmark codes (to run baseline with Mixup, using the --mixup option)
isResNet = False# True for ResNet-18 for Cifar10; otherwise a simple CNN for MNIST

if isResNet:
    currentModel ="Cifar10"
else:
    currentModel="MNIST"
                
BATCH_SIZE = 128
CLASS_NUM = 10
  
if isResNet:
 IMAGE_SIZE = 32 #for Cifar
else:
 IMAGE_SIZE = 28 #for MNIST

LR_SCHEDULE = [(0, 0.1), (100, 0.01), (150, 0.001)] 
WEIGHT_DECAY = 1e-4
FILTER_SIZES = [64, 128, 256, 512]
MODULE_SIZES = [2, 2, 2, 2]


#----------------For MNIST---------------------------------
def maybe_download(url, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    filename = url.split('/')[-1]
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        logger.info("Downloading to {}...".format(filepath))
        download(url, work_directory)
    return filepath

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        data = data.astype('float32') / 255.0
        return data

def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return labels

class Mnist(RNGDataFlow):
    """
    Produces [image, label] in MNIST dataset,
    image is 28x28 in the range [0,1], label is an int.
    """
    
    DIR_NAME = 'mnist_data' 
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    def __init__(self, train_or_test, shuffle=True, dir=None):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        if dir is None:
            dir = get_dataset_path(self.DIR_NAME)
        assert train_or_test in ['train', 'test']
        self.train_or_test = train_or_test
        self.shuffle = shuffle

        def get_images_and_labels(image_file, label_file):
            f = maybe_download(self.SOURCE_URL + image_file, dir)
            images = extract_images(f)
            f = maybe_download(self.SOURCE_URL + label_file, dir)
            labels = extract_labels(f)
            assert images.shape[0] == labels.shape[0]
            ids, other = zip(*((id, other) for id, other in zip(images, labels) if other !=-7))

            images=np.array(ids)
            labels =np.array(other)
            return images, labels

        if self.train_or_test == 'train':
            self.images, self.labels = get_images_and_labels(
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz')
        else:
            self.images, self.labels = get_images_and_labels(
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz')

    def size(self):
        return self.images.shape[0]

    def get_data(self):
        idxs = list(range(self.size()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img = self.images[k].reshape((28, 28))
            label = self.labels[k]
            yield [img, label]


            
#----------------For ResNet---------------------------------
def preactivation_block(input, num_filters, stride=1):
    num_filters_in = input.get_shape().as_list()[1]

    # residual
    net = BNReLU(input)
    residual = Conv2D('conv1', net, num_filters, kernel_size=3, strides=stride, use_bias=False, activation=BNReLU)
    residual = Conv2D('conv2', residual, num_filters, kernel_size=3, strides=1, use_bias=False)

    # identity
    shortcut = input
    if stride != 1 or num_filters_in != num_filters:
        shortcut = Conv2D('shortcut', net, num_filters, kernel_size=1, strides=stride, use_bias=False)

    return shortcut + residual

class ResNet_Cifar_AdaMixup(ModelDesc):
    def inputs(self):
        # a pair of images as inputs
        if isResNet:
         return [tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE,3], 'input'),
                tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE,3], 'input_2'),
                tf.placeholder(tf.float32, [None, CLASS_NUM], 'label'),
                tf.placeholder(tf.float32, [None, CLASS_NUM], 'label_2')]
        else:
         return [tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE], 'input'),
                tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE], 'input_2'),
                tf.placeholder(tf.float32, [None, CLASS_NUM], 'label'),
                tf.placeholder(tf.float32, [None, CLASS_NUM], 'label_2')]

    def build_graph(self, image_1, image_2,label_1,label_2): 
        if get_current_tower_context().is_training:
            # construct images for mixing and images as unseeen for Intrusion Discriminator 
            image_1_f,image_1_s = tf.split(image_1,2,0)
            image_2_f,image_2_s = tf.split(image_2,2,0)
            label_1_f,label_1_s = tf.split(label_1,2,0)
            label_2_f,label_2_s = tf.split(label_2,2,0)

            image_1 = image_1_f
            image_2 = image_2_f
            label_1 = label_1_f
            label_2 = label_2_f

            image_unseen = tf.concat([image_1_s],0,name="unseen_batch_images")
            label_unseen = tf.concat([label_1_s],0,name="unseen_batch_labels")
            
        else:
            image_unseen = image_1
            label_unseen = label_1

        assert tf.test.is_gpu_available()

        if get_current_tower_context().is_training:  

            #generate images and features for Policy Generator
            m1=tf.truncated_normal(tf.shape(image_1),mean=0,stddev=1,dtype=tf.float32,name='avegareImage_rate1')
            m2=tf.truncated_normal(tf.shape(image_2),mean=0,stddev=1,dtype=tf.float32,name='avegareImage_rate2')            
            image_gate = tf.div((m1*image_1+m2*image_2),2*1.0,"dyn_gate")            

            #using the same network architecture as that for Mixing and Intrusion Discriminator
            if isResNet:
             pytorch_default_init = tf.variance_scaling_initializer(scale=1.0 / 3, mode='fan_in', distribution='uniform')
             with argscope([Conv2D, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
                argscope(Conv2D, kernel_initializer=pytorch_default_init):
              net = Conv2D('conv0', image_gate, 64, kernel_size=3, strides=1, use_bias=False)
              for i, blocks_in_module in enumerate(MODULE_SIZES):
                for j in range(blocks_in_module):
                    stride = 2 if j == 0 and i > 0 else 1
                    with tf.variable_scope("res%d.%d" % (i, j)):
                        net = preactivation_block(net, FILTER_SIZES[i], stride)
              fc1 = GlobalAvgPooling('gap', net)
              gate= fc1
            else: #simple ConvNet
             image_gate = tf.expand_dims(image_gate * 2 - 1, 3) 
             with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):
              c0 = Conv2D('conv0', image_gate)
              p0 = MaxPooling('pool0', c0, 2)
              c1 = Conv2D('conv1', p0)
              c2 = Conv2D('conv2', c1)
              p1 = MaxPooling('pool1', c2, 2)
              c3 = Conv2D('conv3', p1)
              fc1 = FullyConnected('fc0', c3, 512, nl=tf.nn.relu)
              fc1 = Dropout('dropout', fc1, 0.5)
              gate= fc1
                
            gate_2 = FullyConnected('gate_linear', gate, 100,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))
            gate_3 = FullyConnected('gate_linear2', gate_2, 3,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))

            #generate the interval values using a softmax function/discard the last element
            cc = tf.nn.softmax(gate_3,1)
            a,b,c=tf.split(cc,3,1)
            add_tensor_summary(a, ['histogram', 'rms', 'sparsity'], name='a')
            add_tensor_summary(b, ['histogram', 'rms', 'sparsity'], name='b')
            add_tensor_summary(c, ['histogram', 'rms', 'sparsity'], name='c')
            
            alpha = a           
            add_tensor_summary(alpha, ['histogram', 'rms', 'sparsity'], name='alpha')
            alpha2 = b
            add_tensor_summary(alpha2, ['histogram', 'rms', 'sparsity'], name='alpha2')

            #reparameterization trick
            #lent = BATCH_SIZE/2
            uni = tf.random_uniform([BATCH_SIZE/2,1], 0, 1) 
            ran = alpha + uni*alpha2
            #end of reparameterization trick
           
            weight = ran
            
            
        if get_current_tower_context().is_training:

            #generate mixed images using the policy region generated above
            add_tensor_summary(weight, ['histogram', 'rms', 'sparsity'], name='weight_generated_from_uniformTimesA')

            if isResNet:
              x_weight = tf.reshape(weight,[BATCH_SIZE/2, 1, 1,1])
            else:
              x_weight = tf.reshape(weight,[BATCH_SIZE/2, 1, 1])

            y_weight = tf.reshape(weight,[BATCH_SIZE/2, 1])
            
            image_mix = image_1 * x_weight + image_2 * (1 - x_weight)
            label_mix = label_1 * y_weight + label_2 * (1 - y_weight)

            #adding mixed images into the original training data set, forming a new training data for network training
            allimage =  tf.concat([image_mix,image_1],0,name="all_images")
            alllabel = tf.concat([label_mix,label_1],0,'all_labels')
            
            idx = tf.random_shuffle(tf.range(tf.shape(allimage)[0]))                 
            image = tf.gather(allimage,idx)
            label_mix = tf.gather(alllabel,idx)

            add_tensor_summary(label_mix, ['histogram', 'rms', 'sparsity'], name='label_mix')

        else:
            image_mix = image_1
            image = image_1
            label_mix = label_1
            image_mix = tf.expand_dims(image_mix * 2 - 1, 3)
            

        def share_mixup_net(image):
          if isResNet:  
           MEAN_IMAGE = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
           STD_IMAGE = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
           image = ((image / 255.0) - MEAN_IMAGE) / STD_IMAGE
           image = tf.transpose(image, [0, 3, 1, 2])

          if isResNet: #ResNet
           pytorch_default_init = tf.variance_scaling_initializer(scale=1.0 / 3, mode='fan_in', distribution='uniform')
           with argscope([Conv2D, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
                argscope(Conv2D, kernel_initializer=pytorch_default_init):
            net = Conv2D('conv0', image, 64, kernel_size=3, strides=1, use_bias=False)
            for i, blocks_in_module in enumerate(MODULE_SIZES):
                for j in range(blocks_in_module):
                    stride = 2 if j == 0 and i > 0 else 1
                    with tf.variable_scope("res%d.%d" % (i, j)):
                        net = preactivation_block(net, FILTER_SIZES[i], stride)
            fc1 = GlobalAvgPooling('gap', net)

          else: #simple ConvNet
           image = tf.expand_dims(image * 2 - 1, 3)
           with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):
            c0 = Conv2D('conv0', image)
            p0 = MaxPooling('pool0', c0, 2)
            c1 = Conv2D('conv1', p0)
            c2 = Conv2D('conv2', c1)
            p1 = MaxPooling('pool1', c2, 2)
            c3 = Conv2D('conv3', p1)
            fc1 = FullyConnected('fc0', c3, 512, nl=tf.nn.relu)
            fc1 = Dropout('dropout', fc1, 0.5)

          return fc1

        with tf.variable_scope("mixup_model") as scope:
            net_mixup = share_mixup_net(image)
            
            scope.reuse_variables()

            #forming the binary training data set for Intrusion Discriminator
            if get_current_tower_context().is_training:
                 #reshape images
                 image_unseen = tf.squeeze(image_unseen)
                 image_1 = tf.squeeze(image_1)
                 image_2 = tf.squeeze(image_2)
                 image = tf.squeeze(image)
                 image_mix = tf.squeeze(image_mix)

                 #generate negative and positive images
                 binary_images_pos = tf.concat([image_unseen,image_1],0,'bniary_examples_pos')
                 binary_images_neg = tf.concat([image_mix],0,'bniary_examples_neg')

                 #assign negative and positive labels
                 num_samples = tf.shape(binary_images_pos)[0]                 
                 indices = tf.ones([num_samples], tf.int32)
                 depth = 2
                 labels_pos = tf.one_hot(indices, depth)
                 
                 num_samples = tf.shape(binary_images_neg)[0]                 
                 indices = tf.zeros([num_samples], tf.int32)
                 depth = 2
                 labels_neg = tf.one_hot(indices, depth)

                 #forming training data set for Intrusion Discriminator
                 binary_images = tf.concat([binary_images_pos,binary_images_neg],0,'binary_examples')                 
                 binary_labels = tf.concat([labels_pos,labels_neg],0,'binary_labels')

                 #reshuttle training dataset
                 num_samples = tf.shape(binary_images)[0]
                 
                 idx = tf.random_shuffle(tf.range(num_samples))
                 binary_images = tf.gather(binary_images,idx)
                 binary_labels = tf.gather(binary_labels,idx)

                 #train Intrusion Discriminator
                 net_discrimintor = share_mixup_net(binary_images)
                 
        #compute loss for both the Classifier networks and the Intrusion Discriminator networks         
        logits = FullyConnected('linear', net_mixup, CLASS_NUM,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))
        ce_cost = tf.nn.softmax_cross_entropy_with_logits(labels=label_mix, logits=logits)
        ce_cost1 = tf.reduce_mean(ce_cost, name='cross_entropy_loss_mixup')
        add_tensor_summary(ce_cost1, ['histogram', 'rms', 'sparsity'], name='cross_entropy_loss_mixupNetwork')
        
        #calculate the Intrusion Discriminator loss
        if get_current_tower_context().is_training:
          extra = FullyConnected('linear001', net_discrimintor, CLASS_NUM,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))
          logits_2 = FullyConnected('linear2', extra, 2,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))

          ce_cost = tf.nn.softmax_cross_entropy_with_logits(labels=binary_labels, logits=logits_2)
          ce_cost2 = tf.reduce_mean(ce_cost, name='cross_entropy_loss_intrusion_discriminator')
          add_tensor_summary(ce_cost2, ['histogram', 'rms', 'sparsity'], name='cross_entropy_loss_intrusion_discriminator')

          #sum up of two losses in training: Classifier + Intrusion Discriminator
          ce_cost = tf.div((ce_cost1+ce_cost2),1*1.0,"cross_entropy_loss")#harry #God
        else:
          ce_cost =ce_cost1

        #compute accuracy    
        if get_current_tower_context().is_training:
            single_label = tf.to_int32(tf.argmax(label_mix, axis=1))
        else:          
            single_label = tf.to_int32(tf.argmax(label_1, axis=1))

        logits_merge = logits
        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits_merge, single_label, 1)), name='wrong_vector')
        err = tf.reduce_mean(wrong, name='train_error')

        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'), ce_cost)
        add_param_summary(('.*/W', ['histogram']))


        # weight decay on all W matrixes. including convolutional layers
        wd_cost = tf.multiply(WEIGHT_DECAY, regularize_cost('.*', tf.nn.l2_loss), name='wd_cost')
        add_tensor_summary(wd_cost, ['histogram', 'rms', 'sparsity'], name='WEIGHT_DECAY_cost')

        return tf.add_n([ce_cost, wd_cost], name='cost')
        
    def optimizer(self): 
      if isResNet:  #optimizer used for ResNet
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt
      else:   #optimizer used for MNSIT 
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 30,
            decay_rate=0.3, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        opt = tf.train.AdamOptimizer(lr)
        return opt

def get_data_adaMixup(train_or_test, isMixup, alpha):
    isTrain = train_or_test == 'train'
    if isResNet:
     ds = dataset.Cifar10(train_or_test)
    else:
     ds =Mnist(train_or_test)
        
    if isResNet:
     if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
        ]
        ds = AugmentImageComponent(ds, augmentors)

    batch = BATCH_SIZE
    ds = BatchData(ds, batch, remainder=not isTrain)

    def f(dp):

        images, labels = dp
        num_samples = labels.shape[0] 
        one_hot_labels = np.eye(CLASS_NUM)[labels]  # one hot coding

        index = np.random.permutation(num_samples)
        if isTrain:
          x1, x2 = images, images[index]
          y1, y2 = one_hot_labels, one_hot_labels[index]
        else:
          x1, x2 = images, images
          y1, y2 = one_hot_labels, one_hot_labels
        return [x1,x2,y1,y2]

    ds = MapData(ds, f)
    return ds


class ResNet_Cifar_baseline(ModelDesc):
    def inputs(self):
        if isResNet:
          return [tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE,3], 'input'),
                tf.placeholder(tf.float32, [None, CLASS_NUM], 'label')]
        else:
          return [tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE], 'input'),
                tf.placeholder(tf.float32, [None, CLASS_NUM], 'label')]
            
    def build_graph(self, image, label):
        assert tf.test.is_gpu_available()

        if isResNet: 
         MEAN_IMAGE = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
         STD_IMAGE = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
         image = ((image / 255.0) - MEAN_IMAGE) / STD_IMAGE
         image = tf.transpose(image, [0, 3, 1, 2])

        if isResNet: #network for ResNet
         pytorch_default_init = tf.variance_scaling_initializer(scale=1.0 / 3, mode='fan_in', distribution='uniform')
         with argscope([Conv2D, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
                argscope(Conv2D, kernel_initializer=pytorch_default_init):
            net = Conv2D('conv0', image, 64, kernel_size=3, strides=1, use_bias=False)
            for i, blocks_in_module in enumerate(MODULE_SIZES):
                for j in range(blocks_in_module):
                    stride = 2 if j == 0 and i > 0 else 1
                    with tf.variable_scope("res%d.%d" % (i, j)):
                        net = preactivation_block(net, FILTER_SIZES[i], stride)
            net = GlobalAvgPooling('gap', net)
            logits = FullyConnected('linear', net, CLASS_NUM,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))
        else: #CNN for MNIST 
         image = tf.expand_dims(image * 2 - 1, 3) 
         with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):
            c0 = Conv2D('conv0', image)
            p0 = MaxPooling('pool0', c0, 2)
            c1 = Conv2D('conv1', p0)
            c2 = Conv2D('conv2', c1)
            p1 = MaxPooling('pool1', c2, 2)
            c3 = Conv2D('conv3', p1)
            fc1 = FullyConnected('fc0', c3, 512, nl=tf.nn.relu)
            fc1 = Dropout('dropout', fc1, 0.5)
            logits = FullyConnected('fc1', fc1, out_dim=10, nl=tf.identity)
        
        ce_cost = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
        ce_cost = tf.reduce_mean(ce_cost, name='cross_entropy_loss')

        single_label = tf.to_int32(tf.argmax(label, axis=1))
        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, single_label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'), ce_cost)
        add_param_summary(('.*/W', ['histogram']))

        # weight decay on all W matrixes. including convolutional layers
        wd_cost = tf.multiply(WEIGHT_DECAY, regularize_cost('.*', tf.nn.l2_loss), name='wd_cost')

        return tf.add_n([ce_cost, wd_cost], name='cost')

    def optimizer(self):
      if isResNet:  #optimizer for ResNet
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt
      else: #optimizer for CNN
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)
          

def get_data_ResNet(train_or_test, isMixup, alpha):
    isTrain = train_or_test == 'train'
    if isResNet:
     ds = dataset.Cifar10(train_or_test)
    else:
     ds =Mnist(train_or_test)

    if isResNet: #data augmentation for ResNet
     if isTrain:
         
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
        ]
        ds = AugmentImageComponent(ds, augmentors)
    batch = BATCH_SIZE
    ds = BatchData(ds, batch, remainder=not isTrain)

    def f(dp):
        images, labels = dp
        one_hot_labels = np.eye(CLASS_NUM)[labels]  # one hot coding
        if not isTrain or not isMixup:
            return [images, one_hot_labels]

        # mixup implementation:

        weight = np.random.beta(alpha, alpha, BATCH_SIZE)

        if isResNet: 
              x_weight = weight.reshape(BATCH_SIZE, 1, 1, 1)
        else: 
              x_weight = weight.reshape(BATCH_SIZE, 1, 1)
        
        y_weight = weight.reshape(BATCH_SIZE, 1)
        index = np.random.permutation(BATCH_SIZE)

        x1, x2 = images, images[index]
        x = x1 * x_weight + x2 * (1 - x_weight)
        y1, y2 = one_hot_labels, one_hot_labels[index]
        y = y1 * y_weight + y2 * (1 - y_weight)
        return [x, y]

    ds = MapData(ds, f)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--mixup', help='enable mixup', action='store_true')
    parser.add_argument('--alpha', default=1, type=float, help='alpha in mixup')
    
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if isAdaMixup:        
      log_folder = 'train_log/AdaMixup_%s_' % currentModel
    else:
      log_folder = 'train_log/Baseline_%s_' % currentModel
        
    logger.set_logger_dir(os.path.join(log_folder))

    if isAdaMixup:
        dataset_train = get_data_adaMixup('train', args.mixup, args.alpha)
        dataset_test = get_data_adaMixup('test', args.mixup, args.alpha)
    else:
        dataset_train = get_data_ResNet('train', args.mixup, args.alpha)
        dataset_test = get_data_ResNet('test', args.mixup, args.alpha)
    
    steps_per_epoch = dataset_train.size()
    if isAdaMixup: #AdaMixup
     if isResNet:
      config = TrainConfig(
        model=ResNet_Cifar_AdaMixup(),
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate', LR_SCHEDULE) #reschedule learning rate
        ],
        max_epoch=1400,
        steps_per_epoch=steps_per_epoch,
        session_init=SaverRestore(args.load) if args.load else None
      )
     else:
      config = TrainConfig(
        model=ResNet_Cifar_AdaMixup(),
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
        ],
        max_epoch=400,
        steps_per_epoch=steps_per_epoch,
        session_init=SaverRestore(args.load) if args.load else None
      )
    else: #the benchmarking codes
     if isResNet:
      config = TrainConfig(
        model=ResNet_Cifar_baseline(), 
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate', LR_SCHEDULE)#reschedule learning rate
        ],
        max_epoch=400,
        steps_per_epoch=steps_per_epoch,
        session_init=SaverRestore(args.load) if args.load else None
      )
     else:
      config = TrainConfig(
        model=ResNet_Cifar_baseline(),
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
        ],
        max_epoch=400,
        steps_per_epoch=steps_per_epoch,
        session_init=SaverRestore(args.load) if args.load else None
      )
        
      
    launch_train_with_config(config, SimpleTrainer())
