#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import tensorflow as tf
import random
import settings


class Model(object):
    """最简单的rnn编码器-解码器模型"""

    def __init__(self, vocab_size=4000, buckets=[(140,30)], size=128,
                 num_layers=3, max_gradient_norm=5.0, batch_size=512, learning_rate = 0.5,
                 learning_rate_decay_factor = 0.99, cell_type = 'GRU',
                 num_samples = 512, forward_only=False):
        self.vocab_size = vocab_size
        self.buckets = buckets
        self.size = size
        self.num_layers = num_layers
        self.max_gradient_norm = max_gradient_norm
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.cell_type = cell_type
        self.num_samples = num_samples
        self.forward_only = forward_only
        self.global_step = tf.Variable(0, trainable=False)

        # sampled softmax
        output_projection = None
        softmax_loss_function = None
        if self.num_samples > 0 and self.num_samples < self.vocab_size:
            w = tf.get_variable("proj_w", [self.size, self.vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, self.num_samples,
                self.vocab_size)
            softmax_loss_function = sampled_loss

        #create multilayer rnn
        single_cell = tf.nn.rnn_cell.GRUCell(self.size)
        if self.cell_type == 'LSTM':
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        cell = single_cell
        if self.num_layers >1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)

        #seq2seq
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                          num_encoder_symbols = self.vocab_size, num_decoder_symbols = self.vocab_size,
                          embedding_size = size, output_projection=output_projection,
                          feed_previous=do_decode)

        #feeds for input
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(self.buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
        for i in xrange(self.buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))
        #预测decoder的后一个字
        targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

        #训练输出和loss
        if forward_only:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, targets, self.target_weights,
                       self.buckets, lambda x,y:seq2seq_f(x,y,True), softmax_loss_function = softmax_loss_function,
                       per_example_loss=False, name=None)
            #output projection
            if output_projection is not None:
                for b in xrange(len(self.buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets( self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, self.buckets,lambda x, y: seq2seq_f(x, y, False),softmax_loss_function=softmax_loss_function)

        #梯度下降
        params = tf.trainable_variables()
        if not forward_only:
             self.gradient_norms = []
             self.updates = []
             opt = tf.train.GradientDescentOptimizer(self.learning_rate)
             for b in xrange(len(self.buckets)):
                 gradients = tf.gradients(self.losses[b], params)
                 clipped_gradients, norm = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
                 self.gradient_norms.append(norm)
                 self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
            bucket_id, forward_only):
            """训练模型的一步

            Args:
            session: tensorflow session to use.
            encoder_inputs: list of numpy int vectors to feed as encoder inputs.
            decoder_inputs: list of numpy int vectors to feed as decoder inputs.
            target_weights: list of numpy float vectors to feed as target weights.
            bucket_id: which bucket of the model to use.
            forward_only: whether to do the backward step or only forward.

            Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.

            Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
            """

            # 检查数据size
            encoder_size, decoder_size = self.buckets[bucket_id]
            if len(encoder_inputs) != encoder_size:
                raise ValueError("Encoder length must be equal to the one in bucket,"
                    " %d != %d." % (len(encoder_inputs), encoder_size))
            if len(decoder_inputs) != decoder_size:
                raise ValueError("Decoder length must be equal to the one in bucket,"
                    " %d != %d." % (len(decoder_inputs), decoder_size))
            if len(target_weights) != decoder_size:
                raise ValueError("Weights length must be equal to the one in bucket,"
                    " %d != %d." % (len(target_weights), decoder_size))

            # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
            input_feed = {}
            for l in xrange(encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            for l in xrange(decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]

            # Since our targets are decoder inputs shifted by one, we need one more.
            last_target = self.decoder_inputs[decoder_size].name
            input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

            # Output feed: depends on whether we do a backward step or not.
            if not forward_only:
                output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                    self.gradient_norms[bucket_id],  # Gradient norm.
                    self.losses[bucket_id]]  # Loss for this batch.
            else:
                output_feed = [self.losses[bucket_id]]  # Loss for this batch.
                for l in xrange(decoder_size):  # Output logits.
                    output_feed.append(self.outputs[bucket_id][l])

            outputs = session.run(output_feed, input_feed)
            if not forward_only:
                return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
            else:
                return None, outputs[0], outputs[1:] # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
        data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
        bucket_id: integer, which bucket to get the batch for.

        Returns:
        The triple (encoder_inputs, decoder_inputs, target_weights) for
        the constructed batch that has the proper format to call step(...) later.
        """

        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                for batch_idx in xrange(self.batch_size)], dtype=np.int32)
            )

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                for batch_idx in xrange(self.batch_size)], dtype=np.int32)
            )

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == settings.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
