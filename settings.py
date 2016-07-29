#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf

"""数据"""
data_dir='data/test/'
sr_vocab_size=4000
tg_vocab_size=4000

"""模型中预处理"""
PAD = '__PAD__'
GO = '__GO__'
EOS = '__EOS__'
UNK = '__UNK__'
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
#buckets=[(100,20),(100,30),(110,20),(110,30),(120,20),(120,30),(130,20),(140,30)]
buckets=[(5,5)]
reverse_source = True

"""模型参数"""
attention = True
size=512
#embedding_size = 16
#hidden_size = 64
num_layers = 1
cell_type = 'GRU'
batch_size = 1
num_samples = 512
forward_only = False

"""Adam优化算法参数"""
learning_rate = 0.5
beta1 = 0.9
beta2 = 0.99
epsilon = 0.5
max_gradient_norm = 5.0

"""SGD"""
learning_rate_decay_factor=0.95


"""训练参数"""
train_dir = 'train'
initialize_embedding = False
initialize_embedding_data_num = 100000
steps_per_checkpoint = 1
