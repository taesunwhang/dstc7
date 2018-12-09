import tensorflow as tf
from utils import stack_bidirectional_rnn

class DualEncoder(object):

    def __init__(self, hparams, dropout_keep_prob_ph,
                 context_embedded, context_len, utterances_embedded, utterances_len):

        self.hparams = hparams
        self.dropout_keep_prob = dropout_keep_prob_ph

        self.batch_size = tf.shape(context_embedded)[0]
        self.num_candidates = tf.divide(tf.shape(utterances_embedded)[0], self.batch_size)
        self.max_utterances_len = tf.shape(utterances_embedded)[1]

        self.context_outputs = self.context_encoder(context_embedded, context_len)
        self.utterances_outputs = self.utterance_encoder(utterances_embedded, utterances_len)

    def context_encoder(self, context_embedded, context_len):

        with tf.variable_scope("context-encoder"):
            context_outputs = stack_bidirectional_rnn(
                cell="CUDNNGRU",
                num_layers=self.hparams.rnn_depth,
                num_units=self.hparams.rnn_hidden_dim * 2,
                inputs=context_embedded,
                sequence_length=context_len,
                state_merge="concat",
                output_dropout_keep_prob=self.dropout_keep_prob,
                residual=self.hparams.rnn_depth > 1
            )

        return context_outputs

    def utterance_encoder(self, utterances_embedded, utterances_len):

        # [10,10,281,300]
        with tf.variable_scope("utterances-encoder"):

            utterances_len = tf.reshape(utterances_len, [-1])
            utterances_outputs = stack_bidirectional_rnn(
                cell="CUDNNGRU",
                num_layers=self.hparams.rnn_depth,
                num_units=self.hparams.rnn_hidden_dim * 2,
                inputs=utterances_embedded,
                sequence_length=utterances_len,
                state_merge="concat",
                output_dropout_keep_prob=self.dropout_keep_prob,
                residual=self.hparams.rnn_depth > 1
            )

            # [batch_size, num_candidates, max_utter_len, rnn_hidden_dim * 2]
            utterances_outputs = \
                tf.reshape(utterances_outputs,
                           [self.batch_size, -1, self.max_utterances_len, self.hparams.rnn_hidden_dim * 2])

        return utterances_outputs