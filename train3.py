#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import data_utils
import seq2seq_model
import settings
import tensorflow as tf
import numpy as np
import time
import utils
import os
import math


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
        settings.sr_vocab_size, settings.tg_vocab_size, settings.buckets,
        settings.size, settings.num_layers, settings.max_gradient_norm, settings.batch_size,
        settings.optimizer,use_lstm=settings.use_LSTM, num_samples = settings.num_samples, forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(settings.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

def train():
    print "Preparing data in %s" % settings.data_dir
    sr_train_ids_path, tg_train_ids_path,sr_dev_ids_path, tg_dev_ids_path,sr_vocab_path, tg_vocab_path = data_utils.prepare_data(settings.data_dir)
    print "Reading training data from %s" % settings.data_dir
    train_set = data_utils.read_data(sr_train_ids_path,tg_train_ids_path,settings.max_train_num)
    train_batches,train_bucket_ids = data_utils.batchize(train_set)
    print "Reading development data from %s" % settings.data_dir
    dev_set = data_utils.read_data(sr_dev_ids_path,tg_dev_ids_path)
    dev_batches,dev_bucket_ids = data_utils.batchize(dev_set)

    log_file = open(settings.train_dir+'log.txt','w')
    log_file.write('epoch\tstep\ttime\ttrain-ppx\tdev-ppx\n')
    log_file.flush()

    with tf.Session() as sess:
        print("Creating %d layers of %d units." %
              (settings.num_layers, settings.size))
        model = create_model(sess, False)
        current_epoch,current_step,train_loss = 0,0,0.0
        start_time = time.time()
        while True:
            current_epoch+=1
            for batch_id in xrange(len(train_batches)):
                current_step+=1
                encoder_inputs, decoder_inputs, target_weights = model.preprocess_batch(train_batches[batch_id], train_bucket_ids[batch_id])
                _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, train_bucket_ids[batch_id], False)
                train_loss+=step_loss/settings.steps_per_checkpoint
                if current_step % settings.steps_per_checkpoint == 0:
                    # evaluate in training set
                    train_ppx = math.exp(train_loss)/model.batch_size if train_loss < 300 else float('inf')
                    # evaluate in development set
                    dev_loss=0.0
                    for dev_batch_id in xrange(len(dev_batches)):
                        encoder_inputs, decoder_inputs, target_weights = model.preprocess_batch(dev_batches[dev_batch_id], dev_bucket_ids[dev_batch_id])
                        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, dev_bucket_ids[dev_batch_id], True)
                        dev_loss+=step_loss/len(dev_batches)
                    dev_ppx = math.exp(dev_loss)/model.batch_size if dev_loss < 300 else float('inf')
                    log_file.write("%d\t%d\t%.2f\t%.2f\t%.2f\n" % (current_epoch,model.global_step.eval(),time.time()-start_time,train_ppx,dev_ppx))
                    log_file.flush()
                    sys.stdout.flush()
                    train_loss,dev_loss = 0.0,0.0
                    checkpoint_path = os.path.join(settings.train_dir, "summary.ckpt")
                    model.saver.save(sess, checkpoint_path,global_step=model.global_step)
            train_batches,train_bucket_ids = data_utils.batchize(train_set)

def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()
