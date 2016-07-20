#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf

"""文本数据"""
source_text_file = 'data/LCSTS/PART_I.source.txt'
target_text_file = 'data/LCSTS/PART_I.target.txt'
test_source_text_file = 'data/LCSTS/PART_III.source.txt'
test_target_text_file = 'data/LCSTS/PART_III.target.txt'

"""预处理"""
preprocess_num = 10000
vocab_size = 4000
source_wid_file = 'data/LCSTS/PART_I.source.wid.txt'
target_wid_file = 'data/LCSTS/PART_I.target.wid.txt'
vocab_file = 'data/LCSTS/PART_I.vocab.txt'
inv_vocab_file = 'data/LCSTS/PART_I.invocab.txt'

"""模型中预处理"""
PAD = '__PAD__'
GO = '__GO__'
EOS = '__EOS__'
UNK = '__UNK__'
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
reverse_source = True
add_go_to_target = True
#buckets=[(100,20),(100,30),(110,20),(110,30),(120,20),(120,30),(130,20),(140,30)]
buckets=[(140,30)]

"""模型参数"""
attention = False
size = 16
num_layers = 1
cell_type = 'GRU'
batch_size = 512
num_samples = 0
forward_only = False

"""Adam优化算法参数"""
learning_rate = 0.5
beta1 = 0.9
beta2 = 0.99
epsilon = 0.5
max_gradient_norm = 5.0

"""训练参数"""
train_dir = 'train'
initialize_embedding = False
initialize_embedding_data_num = 100000
max_epoch = 100
