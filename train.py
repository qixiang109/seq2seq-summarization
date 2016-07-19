#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import data_utils
import simple_enc_dec_model
import settings
import tensorflow as tf
import numpy as np
import time
import datetime
import utils
from progressive.bar import Bar
import os

def create_model(session, forward_only):
    model = simple_enc_dec_model.Model(
        vocab_size=len(data_utils.dictionary),
        buckets=settings.buckets,
        size=settings.size,
        num_layers=settings.num_layers,
        max_gradient_norm=settings.max_gradient_norm,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        learning_rate_decay_factor=settings.learning_rate_decay_factor,
        cell_type=settings.cell_type,
        num_samples=settings.num_samples,
        forward_only=forward_only
    )
    ckpt = tf.train.get_checkpoint_state(settings.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Initilizing model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

previous_batch_losses  = []
current_step = 0
def do_batch(model,session,batch_data,bucket_id):
    global current_step,previous_batch_losses
    current_step+=1
    #执行batch
    d1 = datetime.datetime.now()
    encoder_inputs, decoder_inputs, target_weights = model.transpose_batch(batch_data,bucket_id)
    _, step_loss, _ = model.step(session, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
    d2 = datetime.datetime.now()
    #在checkpoint点更新学习率、保存模型
    if current_step%settings.steps_per_checkpoint==0:
        if len(previous_batch_losses)>2 and step_loss > max(previous_batch_losses[-3:]):
            model.learning_rate = model.learning_rate*model.learning_rate_decay_factor
        previous_batch_losses.append(step_loss)
        checkpoint_path = os.path.join(settings.train_dir, "train.ckpt")
        model.saver.save(session, checkpoint_path, global_step=current_step)
    return (step_loss,d2-d1)

current_epoch=0
def do_epoch(model,session):

    """每个epoch重新划分一次batch，然后乱序执行"""
    global current_epoch
    current_epoch+=1
    #每个epoch重新划分一次batch
    batch_datas = []
    batch_bucket_ids = []
    for bucket_id, bucket_data_set in enumerate(data_utils.train_set):
        for one in utils.shuffle_divide(bucket_data_set,settings.batch_size):
            batch_datas.append(one)
            batch_bucket_ids.append(bucket_id)
    batch_num = len(batch_datas)
    #乱序执行batch
    bar = Bar(max_value=len(batch_datas),title='epoch '+str(current_epoch))
    bar.cursor.clear_lines(3)  # Make some room
    bar.cursor.save()  # Mark starting line
    perm = np.random.permutation(batch_num)
    batch_times=[]
    batch_losses=[]
    for batch_id in perm:
        batch_loss, batch_time = do_batch(model,session, batch_datas[batch_id], batch_bucket_ids[batch_id])
        batch_times.append(batch_time)
        batch_losses.append(batch_loss)
        bar.cursor.restore()  # Return cursor to start
        bar.draw(value=len(batch_times),newline=True)  # Draw the bar!
        time_used = np.sum(batch_times)
        time_eta = time_used/len(batch_times) * (len(batch_datas)-len(batch_times))
        print 'elapsed: '+str(time_used)[0:10]+' eta: '+str(time_eta)[0:10]+'\r'
        epoch_loss = np.sum(batch_losses)
        print 'epoch loss: '+str(round(epoch_loss,2))+' batch loss: '+str(round(batch_loss,2))+'\r'
    return np.sum(batch_losses)

def train():
    fw = open(settings.test_decode_file,'w')
    fw.close()
    with tf.Session() as sess:
        print("Creating %d layers of %d units." %
              (settings.num_layers, settings.size))
        model = create_model(sess, settings.forward_only)
        # train loop
        while True:
            loss=0
            loss = do_epoch(model,sess)
            test(sess, model)
            if current_epoch>settings.max_epoch:
                break

def decode(wordlist,session=None,model=None):

    #格式化输入
    source_wids = [data_utils.dictionary[w] if w in data_utils.dictionary else data_utils.dictionary[settings.UNK] for w in wordlist]
    formated_source, formated_target, bucket_id = data_utils.format_source_target(source_wids,[],True)
    #print 'format target', " ".join([data_utils.inv_dictionary[output] for output in formated_target]).encode('utf-8')

    #get batch
    original_batch_size = model.batch_size
    model.batch_size = 1  # We decode one sentence at a time.
    encoder_inputs, decoder_inputs,target_weights = model.transpose_batch([(formated_source, formated_target)],bucket_id)
    _, _, output_logits = model.step(session, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True)
    model.batch_size = original_batch_size
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if settings.EOS_ID in outputs:
        outputs = outputs[:outputs.index(settings.EOS_ID)]
    #if settings.PAD_ID in outputs:
    #    outputs = outputs[:outputs.index(settings.PAD_ID)]
    # Print out French sentence corresponding to outputs.
    return " ".join([data_utils.inv_dictionary[output] for output in outputs]).encode('utf-8')

def test(session, model):
    fw = open(settings.test_decode_file,'a')
    for i,wordlist in enumerate(data_utils.test_data[0:10]):
        fw.write(str(i)+' '+decode(wordlist,session,model).encode('utf-8')+'\n')
        fw.write('\n')
    fw.close()

if __name__ == '__main__':
    data_utils.load_and_prepare_data()
    train()
