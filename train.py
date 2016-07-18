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

current_epoch=0

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


def do_batch(model,session,batch_data,bucket_id):
    d1 = datetime.datetime.now()
    encoder_inputs, decoder_inputs, target_weights = model.transpose_batch(batch_data,bucket_id)
    _, step_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                 target_weights, bucket_id, False)
    d2 = datetime.datetime.now()
    duration = d2-d1
    return (step_loss,duration)

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
        time_used = np.sum(batch_times)
        time_eta = time_used/len(batch_times) * (len(batch_datas)-len(batch_times))
        mean_loss = np.mean(batch_losses)
        bar.cursor.restore()  # Return cursor to start
        bar.draw(value=len(batch_times),newline=True)  # Draw the bar!
        print 'used: '+str(time_used)[0:10]+' eta: '+str(time_eta)[0:10]
        print 'mean loss: '+str(round(mean_loss,2))

    return mean_loss


def train():
    # load and prepare data
    data_utils.load_and_prepare_data()

    # create model
    with tf.Session() as sess:
        print("Creating %d layers of %d units." %
              (settings.num_layers, settings.size))
        model = create_model(sess, settings.forward_only)
        # Read data into buckets and compute their sizes.
        # train loop
        mean_losses=[]
        while True:
            loss = do_epoch(model,sess)
            mean_losses.append(loss)
            if current_epoch<settings.min_epoch:
                continue
            if current_epoch>settings.max_epoch:
                break
            converge = True
            for i in range(1,settings.look_back+1):
                if np.abs(mean_losses[-1*i]-mean_losses[-1*(i+1)]) > settings.convergence:
                    converge=False
                    break
            if converge:
                break

if __name__ == '__main__':
    train()
