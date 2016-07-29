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
        settings.learning_rate, settings.learning_rate_decay_factor,
        forward_only=forward_only)
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
    train_set = data_utils.read_data(sr_train_ids_path,tg_train_ids_path)
    train_batches,train_bucket_ids = data_utils.batchize(train_set)
    print "Reading development data from %s" % settings.data_dir
    dev_set = data_utils.read_data(sr_dev_ids_path,tg_dev_ids_path)
    dev_batches,dev_bucket_ids = data_utils.batchize(dev_set)

    with tf.Session() as sess:
        print("Creating %d layers of %d units." %
              (settings.num_layers, settings.size))
        model = create_model(sess, False)

        current_epoch = 0
        current_step = 0
        train_losses=[]
        train_times=[]
        while True:
            current_epoch+=1
            for batch_id in xrange(len(train_batches)):
                current_step+=1
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = model.preprocess_batch(train_batches[batch_id], train_bucket_ids[batch_id])
                _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, train_bucket_ids[batch_id], False)
                train_losses.append(step_loss)
                train_times.append(time.time() - start_time)
                if current_step % settings.steps_per_checkpoint == 0:
                    mean_train_loss = sum(train_losses)/settings.steps_per_checkpoint
                    mean_train_time = sum(train_times)/settings.steps_per_checkpoint
                    train_ppx = math.exp(mean_train_loss) if mean_train_loss < 300 else float('inf')
                    print "global step %d learning rate %.4f step-time %.2f perplexity %.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                mean_train_time , train_ppx)
                    train_losses=[]
                    train_times=[]
                    # evaluate in development set
                    dev_losses=[]
                    for dev_batch_id in xrange(len(dev_batches)):
                        encoder_inputs, decoder_inputs, target_weights = model.preprocess_batch(dev_batches[dev_batch_id], dev_bucket_ids[dev_batch_id])
                        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, dev_bucket_ids[dev_batch_id], True)
                        dev_losses.append(step_loss)
                    mean_dev_loss = sum(dev_losses)/len(dev_losses)
                    dev_ppx = math.exp(mean_dev_loss) if mean_dev_loss < 300 else float('inf')
                    print "dev perplexity %.2f" % dev_ppx
                    sys.stdout.flush()
                    #checkpoint_path = os.path.join(settings.train_dir, "summary.ckpt")
                    #model.saver.save(sess, checkpoint_path,global_step=model.global_step)
            train_batches,train_bucket_ids = data_utils.batchize(train_set)


def train2():
    print "Preparing data in %s" % settings.data_dir
    sr_train_ids_path, tg_train_ids_path,sr_dev_ids_path, tg_dev_ids_path,sr_vocab_path, tg_vocab_path = data_utils.prepare_data(settings.data_dir)
    print "Reading training data from %s" % settings.data_dir
    train_set = data_utils.read_data(sr_train_ids_path,tg_train_ids_path)
    print "Reading development data from %s" % settings.data_dir
    dev_set = data_utils.read_data(sr_dev_ids_path,tg_dev_ids_path)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(settings.buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

    with tf.Session() as sess:
        print("Creating %d layers of %d units." %
              (settings.num_layers, settings.size))
        model = create_model(sess, False)

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in
            # train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / \
                settings.steps_per_checkpoint
            loss += step_loss / settings.steps_per_checkpoint
            current_step += 1
            if current_step % settings.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                step_time=0.0
                loss=0.0
def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()
