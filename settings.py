#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf

"""数据处理"""
data_dir='data/test/'
sr_vocab_size=4000
tg_vocab_size=4000
PAD = '__PAD__'
GO = '__GO__'
EOS = '__EOS__'
UNK = '__UNK__'
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
buckets=[(142,30)]
reverse_source = True
max_train_num = 500000

"""模型参数"""
attention = True
size=512
num_layers = 1
use_LSTM = False
batch_size = 1
num_samples = 512
forward_only = False

"""优化算法参数"""
adam_optimizer = tf.train.AdamOptimizer(epsilon=1e-8)
adadelta_optimizer = tf.train.AdadeltaOptimizer(epsilon=1e-8)
gd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
optimizer = adam_optimizer
max_gradient_norm = 5.0

"""训练参数"""
train_dir = 'train/'
steps_per_checkpoint = 1
