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
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train():

    # load and prepare data
    data_utils.load_and_prepare_data()

    # create model
    with tf.Session() as sess:
        print("Creating %d layers of %d units." %
              (settings.num_layers, settings.size))
        model = create_model(sess, settings.forward_only)

        # Read data into buckets and compute their sizes.
        train_bucket_sizes = [len(data_utils.train_set[b])
                              for b in xrange(len(settings.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
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
                data_utils.train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            current_step += 1
            print current_step, step_loss


if __name__ == '__main__':
    train()
