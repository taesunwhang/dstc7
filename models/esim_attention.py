import os
import tensorflow as tf

from utils import stack_bidirectional_rnn, sequence_feature

class ESIMAttention(object):

    def __init__(self, hparams, dropout_keep_prob, context_outputs, context_len, utterances_outputs, utterances_len):
        self.hparams = hparams
        self.dropout_keep_prob = dropout_keep_prob

        m_context, m_utterances = self._attention_matching_layer(context_outputs, utterances_outputs)
        m_context_max, m_utterances_max, m_context_fw_last_state, m_utterances_fw_last_state = \
            self._matching_aggregation_layer(m_context, context_len, m_utterances, utterances_len)

        self.context_word_joined_feature = \
            tf.concat(axis=-1, values=[m_context_max, m_utterances_max,
                                       m_context_fw_last_state, m_utterances_fw_last_state])


    def _attention_utterances(self, similarity_matrix, context):
        """

        :param similarity_matrix: [batch_size, num_candidates, response_len, context_len]
        :param context: [batch_size, max_context_len, rnn_hidden_dim*2]
        :return:
        """
        batch_size = tf.shape(similarity_matrix)[0]
        num_candidates = tf.shape(similarity_matrix)[1]
        response_len = tf.shape(similarity_matrix)[2]
        context_len = tf.shape(similarity_matrix)[3]
        dim = tf.shape(context)[-1]

        attention_weight_context = \
            tf.where(tf.equal(similarity_matrix, 0.), similarity_matrix, tf.nn.softmax(similarity_matrix))

        attention_weight_context = tf.reshape(attention_weight_context,
                                              [batch_size, -1, context_len])

        attended_utterances = tf.matmul(attention_weight_context, context)
        attended_utterances = tf.reshape(attended_utterances, [batch_size, num_candidates, response_len, dim])

        return attended_utterances


    def _attention_context(self, similarity_matrix, utterances):
        """

        :param similarity_matrix: [batch_size, num_candidates, response_len, context_len]
        :param utterances: [batch_size, num_candidates, response_len, rnn_hidden_dim*2]
        :param utterances_len: [batch_size, num_candidates]
        :return: attend_context

        """
        similarity_trans_mat = tf.transpose(similarity_matrix, perm=[0, 1, 3, 2])
        attention_weight_response = \
            tf.where(tf.equal(similarity_trans_mat, 0.), similarity_trans_mat, tf.nn.softmax(similarity_trans_mat))

        attended_context = tf.matmul(attention_weight_response, utterances)

        return attended_context


    def _context_response_similarity_matrix(self, context, utterances):
        """

        :param context: bidirectional context_outputs
        :param utterances: bidirectional utterances_outputs
        :return: similarity_matrix
        """

        batch_size = tf.shape(context)[0]
        context_len = tf.shape(context)[1]
        num_candidates = tf.shape(utterances)[1]
        utterances_len = tf.shape(utterances)[2]
        rnn_hidden_dim = tf.shape(utterances)[-1]

        utterances = tf.reshape(utterances, [batch_size, -1, rnn_hidden_dim])
        context = tf.transpose(context, perm=[0, 2, 1])

        similarity_matrix = tf.matmul(utterances, context)
        similarity_matrix = tf.reshape(similarity_matrix, [batch_size, num_candidates, utterances_len, context_len])

        return similarity_matrix

    def _attention_matching_layer(self, context_outputs, utterances_outputs):
        num_candidates = tf.shape(utterances_outputs)[1]

        similarity = self._context_response_similarity_matrix(context_outputs, utterances_outputs)
        attended_context_output = self._attention_context(similarity, utterances_outputs)
        attended_utterances_output = self._attention_utterances(similarity, context_outputs)

        context_outputs = tf.expand_dims(context_outputs, 1)
        context_outputs = tf.tile(context_outputs, [1, num_candidates, 1, 1])

        m_context = tf.concat(axis=-1, values=[context_outputs, attended_context_output,
                                               context_outputs - attended_context_output,
                                               tf.multiply(context_outputs, attended_context_output)])

        m_utterances = tf.concat(axis=-1, values=[utterances_outputs, attended_utterances_output,
                                                  utterances_outputs - attended_utterances_output,
                                                  tf.multiply(utterances_outputs, attended_utterances_output)])

        return m_context, m_utterances

    def _matching_aggregation_layer(self, m_context, context_len, m_utterances, utterances_len):
        batch_size = tf.shape(m_context)[0]
        num_candidates = tf.shape(m_context)[1]
        max_context_len = tf.shape(m_context)[2]
        max_utterances_len = tf.shape(m_utterances)[2]

        # Matching Aggregation Layer
        """m_context"""

        with tf.variable_scope("matching-context"):
            m_context = tf.reshape(m_context,
                                   [-1, max_context_len, self.hparams.rnn_hidden_dim * 8])
            m_context_len = tf.reshape(tf.tile(tf.expand_dims(context_len, -1), [1, num_candidates]), [-1])

            m_context_outputs = stack_bidirectional_rnn(
                cell="CUDNNGRU",
                num_layers=self.hparams.rnn_depth,
                num_units=self.hparams.rnn_hidden_dim * 2,
                inputs=m_context,
                sequence_length=m_context_len,
                state_merge="concat",
                output_dropout_keep_prob=self.dropout_keep_prob,
                residual=self.hparams.rnn_depth > 1
            )
            m_context_fw_bw = tf.split(m_context_outputs, 2, axis=-1)
            value_index = tf.range(batch_size * num_candidates)
            last_index = tf.stack((value_index, m_context_len - 1), axis=1)

            m_context_fw_last_state = tf.gather_nd(m_context_fw_bw[0], last_index)
            m_context_fw_last_state = \
                tf.reshape(m_context_fw_last_state, [batch_size, num_candidates, self.hparams.rnn_hidden_dim])

            m_context_outputs = \
                tf.reshape(m_context_outputs,
                           [batch_size, num_candidates, max_context_len, self.hparams.rnn_hidden_dim * 2])

            m_context_max = tf.reduce_max(m_context_outputs, axis=2)

        """m_utterances"""
        with tf.variable_scope("matching-utterances"):
            m_utterances = tf.reshape(m_utterances, [-1, max_utterances_len, self.hparams.rnn_hidden_dim * 8])
            m_utterances_len = tf.reshape(utterances_len, [-1])

            m_utterances_outputs = stack_bidirectional_rnn(
                cell="CUDNNGRU",
                num_layers=self.hparams.rnn_depth,
                num_units=self.hparams.rnn_hidden_dim * 2,
                inputs=m_utterances,
                sequence_length=m_utterances_len,
                state_merge="concat",
                output_dropout_keep_prob=self.dropout_keep_prob,
                residual=self.hparams.rnn_depth > 1
            )

            m_utterance_fw_bw = tf.split(m_utterances_outputs, 2, axis=-1)
            value_index = tf.range(batch_size * num_candidates)
            last_index = tf.stack((value_index, m_utterances_len - 1), axis=1)

            m_utterances_fw_last_state = tf.gather_nd(m_utterance_fw_bw[0], last_index)
            m_utterances_fw_last_state = \
                tf.reshape(m_utterances_fw_last_state, [batch_size, num_candidates, self.hparams.rnn_hidden_dim])

            m_utterances_outputs = \
                tf.reshape(m_utterances_outputs,
                           [batch_size, num_candidates, max_utterances_len, self.hparams.rnn_hidden_dim * 2])

            m_utterances_max = tf.reduce_max(m_utterances_outputs, axis=2)

        return m_context_max, m_utterances_max, m_context_fw_last_state, m_utterances_fw_last_state
