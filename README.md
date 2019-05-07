# AdaMixUp

This repo contains the code for AdaMixUp, as presented in the paper "MixUp as Locally Linear Out-of-Manifold Regularization", by Hongyu Guo, Yongyi Mao and Richong Zhang, AAAI 2019 (https://arxiv.org/abs/1809.02499)


Requirements and Installation:

1. A computer running macOS or Linux
2. Tensorpack: https://github.com/tensorpack/tensorpack
3. Python version 2.7
4. A Tensorflow installation


Training:

$ CUDA_VISIBLE_DEVICES=0 python GitHub.AdaMixup.AAAI.MNIST.Cifar.py

Acknowledgement:

This reimplementation of AdaMixup is adapted from  https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/cifar10-preact18-mixup.py and https://github.com/tensorpack/tensorpack/blob/master/examples/basics/mnist-convnet.py
