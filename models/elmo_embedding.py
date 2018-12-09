import os
import tensorflow as tf
import numpy as np

from bilm import TokenBatcher, weight_layers, dump_token_embeddings
from bilm.model import BidirectionalLanguageModel

class ELMoEmbeddings(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.vocab_path = self.hparams.word_vocab_path
        self.elmo_options_file = self.hparams.elmo_options_file
        self.elmo_weight_file = self.hparams.elmo_weight_file
        self.token_embedding_file = self.hparams.elmo_token_embedding_file

        self.batcher = TokenBatcher(self.vocab_path)
        if not os.path.exists(self.token_embedding_file):
            print("making dump token embeddings")
            self._make_dump_token_embeddings()
            print("finished making dump_token_embeddings")

    def build_embeddings_op(self, context_ids_ph, utterances_ids_ph, context_sentence_ids_ph):

        bilm = BidirectionalLanguageModel(
            self.elmo_options_file,
            self.elmo_weight_file,
            use_character_inputs=False,
            embedding_weight_file=self.token_embedding_file
        )

        context_emb_op = bilm(context_ids_ph)
        utterances_emb_op = bilm(utterances_ids_ph)
        context_sentence_emb_op = bilm(context_sentence_ids_ph)

        elmo_context_input = weight_layers('input', context_emb_op, l2_coef=0.0)
        with tf.variable_scope('', reuse=True):
            elmo_utterances_input = weight_layers('input', utterances_emb_op, l2_coef=0.0)
            elmo_context_sentence_input = weight_layers('input', context_sentence_emb_op, l2_coef=0.0)

        return (elmo_context_input, elmo_utterances_input, elmo_context_sentence_input)


    def get_toknized_data(self, context_batch, utterances_batch, context_sentence_batch):
        # get nltk tokenized data
        # context, utterances, context_sentence

        #context [None, None] -> okay
        #utterances [None, None, None] -> batch_size * num_candidates, max_utterances_len
        #context_sentence [None, None, None] -> batch_size * max_context_len, max_context_sentence_len

        # batch_size
        context_list = []
        for context in context_batch:
            context_list.append(context[0])

        # batch_size * num_candidates
        utterances_list = []
        for utterances in utterances_batch:
            for response in utterances:
                utterances_list.append(response)

        context_sentence_list = []
        for context_sentences in context_sentence_batch:
            for sentence in context_sentences:
                context_sentence_list.append(sentence)


        context_ids = self.batcher.batch_sentences(context_list)
        utterances_ids = self.batcher.batch_sentences(utterances_list)
        context_sentence_ids = self.batcher.batch_sentences(context_sentence_list)

        return np.array(context_ids), np.array(utterances_ids), np.array(context_sentence_ids)

    def context_sentence_padding(self, elmo_context_sentence_inputs, tot_context_len):
        #elmo_context_sentence_input_val : 39, max_sentence_len, 256
        # [17, 5, 3, 11, 3] -> 17
        max_sentence_len = np.shape(elmo_context_sentence_inputs)[1]
        max_context_len = max(tot_context_len)

        current_index = 0
        length_index = 0

        batch_context_sentence = []
        each_context_sentence = []

        for i in range(len(elmo_context_sentence_inputs)):
            each_context_sentence.append(elmo_context_sentence_inputs[i])
            current_index += 1
            if current_index == tot_context_len[length_index]:
                length_index += 1
                current_index = 0
                batch_context_sentence.append(each_context_sentence)
                each_context_sentence = []
                continue

        pad_context_sentence = []
        for context_sentences in batch_context_sentence:
            if len(context_sentences) < max_context_len:
                padding_value = np.zeros([max_context_len-len(context_sentences),max_sentence_len, 256], np.float32)
                context_sentences = np.concatenate((context_sentences,padding_value), axis=0)

            pad_context_sentence.append(context_sentences)

        return pad_context_sentence
    def _make_dump_token_embeddings(self):

        dump_token_embeddings(
            self.vocab_path, self.elmo_options_file, self.elmo_weight_file, self.token_embedding_file
        )

    def make_placeholders(self):
        context_ids_ph = tf.placeholder(tf.int32, shape=[None,None])
        utterances_ids_ph = tf.placeholder(tf.int32, shape=[None,None])
        context_sentence_ids_ph = tf.placeholder(tf.int32, shape=[None,None])

        return context_ids_ph, utterances_ids_ph, context_sentence_ids_ph
