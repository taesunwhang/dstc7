import tensorflow as tf

from utils import stack_bidirectional_rnn, sequence_feature, positional_encoding, position_encoding_lookup_table

class SentenceBilinearAtt(object):

    def __init__(self, hparams, dropout_keep_prob_ph,
                 context_sentence_embedded, context_sentence_len, tot_context_len,
                 utterances_embedded, utterances_len, speaker):
        """
        :param context_sentence_embedded: [batch_size, max_dialog_len, max_utterances_len, embedding_dim]
        :param utterances_embedded: [batch_size, num_candidates, max_response_length, embedding_dim]
        :param context_sentence_len: [batch_size, context_len] -> each sentence length
        :param tot_context_len: [batch_size] -> each context length(batch by batch)
        :return:
        """

        self.hparams = hparams
        self.dropout_keep_prob = dropout_keep_prob_ph

        self.batch_size = tf.shape(utterances_embedded)[0]
        self.num_candidates = tf.shape(utterances_embedded)[1]
        self.max_response_len = tf.shape(utterances_embedded)[2]

        self.max_dialog_len = tf.shape(context_sentence_embedded)[1]
        self.max_c_sentence_len = tf.shape(context_sentence_embedded)[2]

        self.context_sentence_embedded = context_sentence_embedded
        self.context_sentence_len = context_sentence_len
        self.tot_context_len = tot_context_len
        self.utterances_embedded = utterances_embedded
        self.utterances_len = utterances_len
        self.speaker = speaker

        self.position_embeddings = position_encoding_lookup_table(self.hparams.position_embedding_dim, 10)
        self.pos_emb_bool = False
        self.user_emb_bool = False

    def context_sentence_representation(self, context_sentence_embedded, context_sentence_len, tot_context_len, speaker):

        """context-sentence GLOVE(Avg) representation"""
        context_mask = tf.sequence_mask(context_sentence_len, maxlen=self.max_c_sentence_len)
        context_mask = tf.expand_dims(tf.cast(context_mask, tf.float32), axis=-1)

        masked_context_sentence = tf.multiply(context_sentence_embedded, context_mask)

        context_sentence_sum = tf.reduce_sum(masked_context_sentence, axis=2)

        context_mask = tf.squeeze(context_mask, [-1])

        tot_context_mask = tf.cast(tf.sequence_mask(tot_context_len, maxlen=self.max_dialog_len), tf.float32)

        tot_context_len_tile = tf.tile(tf.expand_dims(tot_context_len, -1), [1, tf.shape(tot_context_mask)[-1]])
        context_sentence_mean = \
            tf.multiply(context_sentence_sum,
                        tf.expand_dims(tf.divide(tot_context_mask, tf.cast(tot_context_len_tile, tf.float32)), -1))

        """context sentence LSTM representation"""
        context_sentence_embedded = tf.reshape(context_sentence_embedded,
                                               [-1, self.max_c_sentence_len, self.hparams.embedding_dim])
        context_sentence_len = tf.reshape(context_sentence_len, [-1])

        with tf.variable_scope("context-sentence-encoder"):
            c_sentence_outputs = stack_bidirectional_rnn(
                cell="CUDNNGRU",
                num_layers=self.hparams.rnn_depth,
                num_units=self.hparams.sentence_rnn_hidden_dim * 2,
                inputs=context_sentence_embedded,
                sequence_length=context_sentence_len,
                state_merge="concat",
                output_dropout_keep_prob=self.dropout_keep_prob,
                residual=self.hparams.rnn_depth > 1
            )

            c_sentence_fw_last_state, c_sentence_bw_first_state = \
                sequence_feature(c_sentence_outputs, context_sentence_len)

            c_sentence_hidden = tf.concat(axis=-1, values=[c_sentence_fw_last_state, c_sentence_bw_first_state])
            c_sentence_hidden = tf.reshape(c_sentence_hidden,
                                           [self.batch_size, self.max_dialog_len, self.hparams.sentence_rnn_hidden_dim*2])

        context_sentence_representation = tf.concat(axis=-1, values=[context_sentence_mean, c_sentence_hidden])

        if self.pos_emb_bool or self.user_emb_bool:
            print("with sentence_features")
            context_w_sentence_feature = self._sentence_order_speaker_feature(context_sentence_representation, speaker)
        else:
            print("without sentence_features")
            context_w_sentence_feature = context_sentence_representation

        return context_w_sentence_feature

    def _sentence_order_speaker_feature(self, context_sentence_representation, speaker):

        position_value = tf.tile(tf.expand_dims(tf.range(tf.shape(speaker)[1]), axis=0),
                                 multiples=[tf.shape(speaker)[0], 1])

        context_sentence_position = positional_encoding(position_value, lookup_table=self.position_embeddings,
                                                        num_units=self.hparams.position_embedding_dim,
                                                        zero_pad=False, scale=False, scope="dialog_sentence_position")
        context_sentence_position = tf.cast(context_sentence_position, tf.float32)

        speaker_one_hot = tf.one_hot(speaker, 2)

        if self.pos_emb_bool and self.user_emb_bool:
            print("position and user")
            context_w_sentence_feature = \
                tf.concat(axis=-1, values=[context_sentence_representation, context_sentence_position, speaker_one_hot])
        elif self.pos_emb_bool and not self.user_emb_bool:
            print("only position")
            context_w_sentence_feature = \
                tf.concat(axis=-1, values=[context_sentence_representation, context_sentence_position])
        elif self.user_emb_bool and not self.pos_emb_bool:
            print("user")
            context_w_sentence_feature = \
                tf.concat(axis=-1, values=[context_sentence_representation, speaker_one_hot])

        projected_context_w_sentence_feature = tf.layers.dense(
            inputs=context_w_sentence_feature,
            units= self.hparams.embedding_dim + self.hparams.sentence_rnn_hidden_dim*2,
            activation=None,
            kernel_initializer=tf.initializers.variance_scaling(
                scale=2.0, mode="fan_in", distribution="normal"),
            name="context_w_sentence_feature_projection"
        )

        return projected_context_w_sentence_feature

    def dialog_representation(self, context_sentence_representation, tot_context_len):

        """context sentence outputs"""
        with tf.variable_scope("tot-context-sentence-layer"):
            tot_context_sentence_outputs = stack_bidirectional_rnn(
                cell="CUDNNGRU",
                num_layers=self.hparams.rnn_depth,
                num_units=self.hparams.sentence_rnn_hidden_dim * 2,
                inputs=context_sentence_representation,
                sequence_length=tot_context_len,
                state_merge="concat",
                output_dropout_keep_prob=self.dropout_keep_prob,
                residual=self.hparams.rnn_depth > 1
            )

            tot_context_fw_last_state, tot_context_bw_first_state = \
                sequence_feature(tot_context_sentence_outputs, tot_context_len)

            tot_context_hidden = tf.concat(axis=-1, values=[tot_context_fw_last_state, tot_context_bw_first_state])


        return tot_context_sentence_outputs, tot_context_hidden

    def response_sentence_representation(self, utterances_embedded, utterances_len):

        """response_sentence_level_encoder"""
        utterances_len = tf.reshape(utterances_len, [self.batch_size, -1])
        response_mask = tf.expand_dims(tf.sequence_mask(utterances_len, maxlen=self.max_response_len), axis=-1)
        response_mask = tf.cast(response_mask, tf.float32)

        # [batch_size, num_candiates, sentence_len, embedding_dim]
        masked_response = tf.multiply(utterances_embedded, response_mask)
        response_sentence_sum = tf.reduce_sum(masked_response, axis=2)
        response_mask = tf.squeeze(response_mask, [-1])
        # [batch_size, num_candidates, 300]
        response_sentence_mean = tf.divide(response_sentence_sum,
                                           tf.expand_dims(tf.reduce_sum(response_mask, axis=-1), axis=-1))

        with tf.variable_scope("response-sentence-encoder"):
            utterances_embedded = tf.reshape(utterances_embedded,
                                             [-1, self.max_response_len, self.hparams.embedding_dim])
            utterances_len = tf.reshape(utterances_len, [-1])
            response_outputs = stack_bidirectional_rnn(
                cell="CUDNNGRU",
                num_layers=self.hparams.rnn_depth,
                num_units=self.hparams.sentence_rnn_hidden_dim * 2,
                inputs=utterances_embedded,
                sequence_length=utterances_len,
                state_merge="concat",
                output_dropout_keep_prob=self.dropout_keep_prob,
                residual=self.hparams.rnn_depth > 1
            )

            response_fw_bw_outputs = tf.split(response_outputs, 2, axis=-1)
            value_index = tf.range(self.batch_size * self.num_candidates)
            first_index = tf.stack((value_index, tf.zeros(tf.shape(value_index), tf.int32)), axis=1)
            last_index = tf.stack((value_index, utterances_len - 1), axis=1)

            r_sentence_fw_last_state = tf.gather_nd(response_fw_bw_outputs[0], last_index)
            r_sentence_bw_last_state = tf.gather_nd(response_fw_bw_outputs[1], first_index)

            r_sentence_hidden = tf.concat(axis=-1, values=[r_sentence_fw_last_state, r_sentence_bw_last_state])
            r_sentence_hidden = \
                tf.reshape(r_sentence_hidden,
                           [self.batch_size, self.num_candidates, self.hparams.sentence_rnn_hidden_dim * 2])

        response_sentence_representation = tf.concat(axis=-1, values=[response_sentence_mean, r_sentence_hidden])

        response_sentence_representation = tf.layers.dense(
            inputs=response_sentence_representation,
            units=self.hparams.sentence_rnn_hidden_dim * 2,
            activation=None,
            kernel_initializer=tf.initializers.variance_scaling(
                scale=2.0, mode="fan_in", distribution="normal"),
            name="response_projection"
        )

        return response_sentence_representation

    def sentence_attention(self, tot_context_att_outputs, response_sentence_representation):

        response_sentence_projection = tf.layers.dense(
            inputs=response_sentence_representation,
            units=self.hparams.sentence_rnn_hidden_dim * 2,
            activation=None,
            kernel_initializer=tf.initializers.variance_scaling(
                scale=2.0, mode="fan_in", distribution="normal"),
            name="response_dialog_bilinear_attention"
        )

        weighted_response = tf.matmul(response_sentence_projection, tf.expand_dims(tot_context_att_outputs, -1))
        softmax_weigthed_response = tf.nn.softmax(tf.squeeze(weighted_response, [-1]), axis=1)

        sentence_attention = tf.multiply(response_sentence_projection,
                                         tf.expand_dims(softmax_weigthed_response, -1))

        return sentence_attention

    def get_sentence_joined_feature(self):
        context_sentence_emb = self.context_sentence_representation(
            self.context_sentence_embedded, self.context_sentence_len, self.tot_context_len, self.speaker)

        tot_context_outputs, tot_context_hidden = self.dialog_representation(context_sentence_emb, self.tot_context_len)

        sentence_attention_joined_feature = self.sentence_attention(
            tot_context_hidden,
            self.response_sentence_representation(self.utterances_embedded, self.utterances_len)
        )

        return sentence_attention_joined_feature
