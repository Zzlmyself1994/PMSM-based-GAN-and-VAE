# -*- coding: utf-8 -*-

"""
Varational Auto Encoder Example.
变分自动编码器
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 基础模块

class Layer:
  def __init__(self, input, n_output):
    self.input = input
    W = tf.Variable(tf.truncated_normal([ int(self.input.get_shape()[1]), n_output ], stddev = 0.001))#tf.shape(input)[0]
    b = tf.Variable(tf.constant(0., shape = [ n_output ]))

    self.raw_output = tf.matmul(input, W) + b
    self.output = tf.nn.relu(self.raw_output)


# 样本集X
n_X = 784 # 28 * 28
n_z = 20 # latent variable count
X = tf.placeholder(tf.float32, shape = [ None, n_X ])

# Encoder

## \mu(X) 采用二层网络
ENCODER_HIDDEN_COUNT = 400
mu = Layer(Layer(X, ENCODER_HIDDEN_COUNT).output, n_z).raw_output

## \Sigma(X) 采用二层网络
log_sigma = Layer(Layer(X, ENCODER_HIDDEN_COUNT).output, n_z).raw_output # 为了训练不出nan? 至少实验的时候，直接让这个网络代表sigma是算不出来的，请高人指点!!!
sigma = tf.exp(log_sigma)

## KLD = D[N(mu(X), sigma(X))||N(0, I)] = 1/2 * sum(sigma_i + mu_i^2 - log(sigma_i) - 1)
KLD = 0.5 * tf.reduce_sum(sigma + tf.pow(mu, 2) - log_sigma - 1, reduction_indices = 1) # reduction_indices = 1代表按照每个样本计算一条KLD


# epsilon = N(0, I) 采样模块
epsilon = tf.random_normal(tf.shape(sigma), name = 'epsilon')

# z = mu + sigma^ 0.5 * epsilon
z = mu + tf.exp(0.5 * log_sigma) * epsilon

# Decoder ||f(z) - X|| ^ 2 重建的X与X的欧式距离，更加成熟的做法是使用crossentropy
def buildDecoderNetwork(z):
  # 构建一个二层神经网络，因为二层神经网络可以逼近任何函数
  DECODER_HIDDEN_COUNT = 400
  layer1 = Layer(z, DECODER_HIDDEN_COUNT)
  layer2 = Layer(layer1.output, n_X)
  return layer2.raw_output

reconstructed_X = buildDecoderNetwork(z)

reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(reconstructed_X, X), reduction_indices = 1)

loss = tf.reduce_mean(reconstruction_loss + KLD)

# minimize loss
n_steps = 100000
learning_rate = 0.01
batch_size = 100

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for step in range(1, n_steps):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([ optimizer, loss ], feed_dict = { X: batch_x })

    if step % 100 == 0:
      print('Step', step, ', Loss:', l)