#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf

# 文本数据
source_text_file = 'data/LCSTS1m.source.txt'
target_text_file = 'data/LCSTS1m.target.txt'

# 预处理
source_wid_file = 'data/LCSTS1m.source.wid.txt'
target_wid_file = 'data/LCSTS1m.target.wid.txt'
vocab_file = 'data/LCSTS1m.vocab.txt'
inv_vocab_file = 'data/LCSTS1m.invvocab.txt'
preprocess_num = 10000
vocab_size = 4000

# 模型中预处理
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
buckets=[(100,20),(100,30),(110,20),(110,30),(120,20),(120,30),(130,20),(140,30)]
dev_ratio = 0.05

# 模型
size = 128
num_layers = 3
cell_type = 'GRU'
batch_size = 1024
learning_rate = 0.5
learning_rate_decay_factor = 0.99
num_samples = 512
max_gradient_norm = 5.0
forward_only = False

#
train_dir = 'train'
steps_per_checkpoint = 100
