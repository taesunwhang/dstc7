import os
import ijson
import nltk
import numpy as np
import random
import threading
import logging
import pickle

# LIMIT_INDEX = 100000
MAX_DIALOG_LENGTH = 10
do_shuffle = True

class DataProcess(object):
    def __init__(self, hparams, data_path, type, word2id=None, char2id=None, test_case=1):

        self.word2id = word2id
        self.char2id = char2id
        if word2id is not None:
            self.id2word = sorted(word2id, key=word2id.__getitem__)
        self.hparams = hparams
        self.data_type = self.hparams.data_type
        self.word_embedding_type = self.hparams.word_embedding_type
        self.input_file_path = data_path
        self.dialog_iter = self.create_dialogue_iter(self.input_file_path, type=type)

        self.index = 0
        self.test_case = test_case

        self.speaker2id = dict()
        if self.data_type == "ubuntu":
            self.speaker2id["participant_1"] = 0
            self.speaker2id["participant_2"] = 1

        elif self.data_type == "advising":
            self.speaker2id["advisor"] = 0
            self.speaker2id["student"] = 1

    def process_dialog(self, dialog, type="train", fr_test_answer_handle=None):
        speakers_l = []
        context_l = []
        utterances_l = []
        candidates_id_l = []
        target_id_l = []

        example_id = dialog['example-id']
        utterances = dialog['messages-so-far']

        context = ""
        speaker = None

        for msg in utterances:
            if speaker is None:
                context += msg['utterance'] + " __eou__ "
                speaker = msg['speaker']
            elif speaker != msg['speaker']:
                context += " __eot__ " + msg['utterance'] + " __eou__ "
                speaker = msg['speaker']
            else:
                context += msg['utterance'] + " __eou__ "
                # same speaker
                continue

            speakers_l.append(speaker)

        context += " __eot__ "
        context_l.append(context)

        if type == "test":
            test_answers = fr_test_answer_handle.readline().split("\t")
            if len(test_answers[1].split(",")) > 1:
                target_id = int(test_answers[1].split(",")[0])
                print("Duplication ! - There are two answers.. regard first value as an answer", target_id)

            else:
                target_id = int(test_answers[1])
            target_index = None

        else:
            correct_answer = dialog['options-for-correct-answers'][0]
            target_id = correct_answer['candidate-id']
            target_index = None

        for i, utterance in enumerate(dialog['options-for-next']):
            if utterance['candidate-id'] == target_id:
                target_index = i
            utterances_l.append(utterance['utterance'] + " __eou__ ")
            candidates_id_l.append(utterance['candidate-id'])

        if target_index is None:
            print('Correct answer not found in options-for-next - example {}. Setting 0 as the correct index'.format(dialog['example-id']))
        else:
            target_id_l.append(target_index)

        return example_id, speakers_l, context_l, utterances_l, target_id_l, candidates_id_l

    def create_dialogue_iter(self, filename, type="train"):

        with open(filename,"rb") as fr_handle:
            fr_test_answer_handle = None
            if type == "test":
                if self.test_case == 1:
                    fr_test_answer_handle = open(self.hparams.test_answer_path, "r", encoding="utf-8")
                elif self.test_case ==2:
                    fr_test_answer_handle = open(self.hparams.test2_answer_path, "r", encoding="utf-8")

                print("Test%d Answer file has opend!" % self.test_case)

            index =0
            json_data = ijson.items(fr_handle,"item")
            for entry in json_data:
                index += 1
                example_id, speakers, context, utterances, target_id, candidates_id = \
                    self.process_dialog(entry, type, fr_test_answer_handle)

                yield (example_id, speakers, context, utterances, target_id, candidates_id)
        if type =="test":
            fr_test_answer_handle.close()

    def tokenize(self, inputs, type="context"):

        tokenized_data_l = []
        tokenized_data_len = []

        # max_inputs_len = 50

        for sentence in inputs:
            tokenized_data = nltk.word_tokenize(sentence)
            # if max_inputs_len < len(tokenized_data) and (type=="response" or type=="dialog_sentence"):
            #     tokenized_data = tokenized_data[0:max_inputs_len]
            tokenized_data_l.append(tokenized_data)
            tokenized_data_len.append(len(tokenized_data))

        return tokenized_data_l, tokenized_data_len

    def get_data_id(self, inputs):
        """
        :param inputs: tokenized (context, utterances)
        :return:
        """
        inputs_id = []
        for sentence in inputs:
            sentences_id = []
            for i, word in enumerate(sentence):
                sentence[i] = word.lower()
                try:
                    word_id = self.word2id[sentence[i]]
                except KeyError:
                    print(sentence[i], ": <UNK>", len(self.word2id))
                    word_id = self.word2id["<UNK>"]
                sentences_id.append(word_id)

            inputs_id.append(sentences_id)

        return inputs_id

    def make_word_to_lower(self, inputs):
        for sentence in inputs:
            for i, word in enumerate(sentence):
                sentence[i] = word.lower()

        return inputs


    def padding_processing(self, batch_data, inputs_length, max_length):
        """
        :param inputs_id: batch_context, batch_utterances
        :param inputs_length: context_len : [batch_size], utterances_len [batch_size,# of candidates]
        :param max_length: maximum_length
        :return:
        """

        # Context
        if np.asarray(inputs_length).ndim == 1:
            for i, input in enumerate(batch_data):
                for sentence in input:
                    if self.word_embedding_type == "elmo":
                        for k in range((max_length - inputs_length[i])):
                            sentence.extend([""])
                    else:
                        sentence.extend([0]*(max_length-inputs_length[i]))

        # Utterances
        else:
            for i, input in enumerate(batch_data):
                for j, sentence in enumerate(input):
                    if self.word_embedding_type == "elmo":
                        for k in range((max_length - inputs_length[i][j])):
                            sentence.extend([""])
                    else:
                        sentence.extend([0]*(max_length-inputs_length[i][j]))
        return batch_data


    def response_negative_sampling(self, num_candidates, utterances, target_id, candidates_id):

        positive_sample = utterances.pop(target_id)
        positive_sample_id = candidates_id.pop(target_id)

        response_ziped = list(zip(utterances, candidates_id))
        random.shuffle(response_ziped)
        negative_samples_id = random.sample(response_ziped, num_candidates - 1)
        neg_samples, neg_candidate_id = zip(*negative_samples_id)
        neg_samples = list(neg_samples)
        neg_candidate_id = list(neg_candidate_id)

        neg_samples.append(positive_sample)
        neg_candidate_id.append(positive_sample_id)


        neg_samples = np.asarray(neg_samples)
        neg_candidate_id = np.asarray(neg_candidate_id)

        # shuffle
        perm = np.arange(num_candidates)
        np.random.shuffle(perm)
        neg_samples = neg_samples[perm]
        neg_candidate_id = neg_candidate_id[perm]


        new_target_id = list(neg_samples).index(positive_sample)

        return neg_samples, [new_target_id], neg_candidate_id

    def word2char(self, sentence, max_word_len=0):
        sentence_char = []
        sentence_char_len = []

        for word_id in list(sentence):
            if word_id == 0:
                sentence_char.append([word_id])
                sentence_char_len.append(0)
            else:
                word_chars = list(self.id2word[word_id])
                word_length = len(word_chars)
                sentence_char_len.append(word_length)
                # max_word_length
                if max_word_len < word_length:
                    max_word_len = word_length

                for word_index, char in enumerate(word_chars):
                    word_chars[word_index] = self.char2id[char]
                sentence_char.append(word_chars)

        return sentence_char, sentence_char_len, max_word_len

    def char_pad_processing(self, sentence_char, max_word_len):
        for chars in sentence_char:
            word_len = len(chars)
            if len(chars) < max_word_len:
                chars.extend([0] * (max_word_len - word_len))
        return sentence_char

    def get_char_batch_data(self, context, utterances):
        max_c_word_len = 0
        max_u_word_len = 0

        char_context = []
        char_context_len = []
        char_context_pad = []

        char_utterances = []
        char_utterances_len = []
        char_utterances_pad = []

        #Context
        for c_batch in context:
            sentence_char, sentence_char_len, max_word_len = self.word2char(c_batch, max_c_word_len)
            max_c_word_len = max_word_len
            char_context.append(sentence_char)
            char_context_len.append(sentence_char_len)

        for c_char_batch in char_context:
            sentence_char_pad = self.char_pad_processing(c_char_batch, max_c_word_len)
            char_context_pad.append(sentence_char_pad)

        char_context_pad = np.asarray(char_context_pad)
        char_context_len = np.asarray(char_context_len)

        #Utterances
        for u_candidates in utterances:
            u_candidates_char = []
            u_candidates_char_len = []
            for response in u_candidates:
                sentence_char, sentence_char_len, max_word_len = self.word2char(response, max_u_word_len)
                max_u_word_len = max_word_len
                u_candidates_char.append(sentence_char)
                u_candidates_char_len.append(sentence_char_len)

            char_utterances.append(u_candidates_char)
            char_utterances_len.append(u_candidates_char_len)

        for u_candidates in char_utterances:
            u_candidates_char_pad = []

            for response_char in u_candidates:
                sentence_char_pad = self.char_pad_processing(response_char, max_u_word_len)
                u_candidates_char_pad.append(sentence_char_pad)

            char_utterances_pad.append(u_candidates_char_pad)

        char_utterances_pad = np.asarray(char_utterances_pad)
        char_utterances_len = np.asarray(char_utterances_len)

        return (char_context_pad, char_context_len), (char_utterances_pad, char_utterances_len)

    def get_context_sentence(self, context):
        context_sentence = context[0].split(" __eot__ ")
        sentences_preprocessing = []
        for i, sentence in enumerate(context_sentence):
            if len(sentence) == 0:
                continue

            sentences_preprocessing.append(sentence.strip())

        # print("dialog utterance original length : ", len(sentences_preprocessing))
        max_dialog_len = MAX_DIALOG_LENGTH
        maximum_sentence_preprocessing = sentences_preprocessing

        if len(sentences_preprocessing) > max_dialog_len:

            maximum_sentence_preprocessing = sentences_preprocessing[-1-max_dialog_len:-1]

        tok_context_s, tok_context_slen = self.tokenize(maximum_sentence_preprocessing)

        return tok_context_s, tok_context_slen

    def get_context_sentence_padding(self,context_sentence_batch, context_sentence_len_batch):
        #maximum
        # for context_sentence
        max_context_sentence = 0
        # for context_sentence_len
        max_context = 0

        context_sentence_pad_batch = context_sentence_batch.copy()
        context_sentence_len_pad_batch = context_sentence_len_batch.copy()

        for context_sentences in context_sentence_batch:
            if max_context < len(context_sentences):
                max_context = len(context_sentences)

            for sentence in context_sentences:
                if max_context_sentence < len(sentence):
                    max_context_sentence = len(sentence)

        #context_sentence
        for context_sentences in context_sentence_pad_batch:
            if len(context_sentences) < max_context:
                for i in range(max_context-len(context_sentences)):
                    if self.word_embedding_type == "elmo":
                        context_sentences.append([""])
                    else:
                        context_sentences.append([0])

            for i, sentence in enumerate(context_sentences):
                if len(sentence) < max_context_sentence:
                    if self.word_embedding_type == "elmo":
                        for k in range(max_context_sentence-len(sentence)):
                            sentence.extend([""])
                    else:
                        sentence.extend([0]*(max_context_sentence-len(sentence)))

        context_len_batch = []
        #context_sentence_len
        for context_sentence_len in context_sentence_len_pad_batch:
            context_len_batch.append(len(context_sentence_len))

            if len(context_sentence_len) < max_context:
                context_sentence_len.extend([0]*(max_context-len(context_sentence_len)))

        return np.asarray(context_sentence_pad_batch), np.asarray(context_sentence_len_pad_batch), \
               np.asarray(context_len_batch)

    def get_context_sentence_padding_elmo(self, context_sentence_batch, context_sentence_len_batch):
        max_context_sentence = 0
        max_context = 0

        context_sentence_pad_batch = context_sentence_batch.copy()
        context_sentence_len_pad_batch = context_sentence_len_batch.copy()

        for context_sentences in context_sentence_batch:
            if max_context < len(context_sentences):
                max_context = len(context_sentences)

            for sentence in context_sentences:
                if max_context_sentence < len(sentence):
                    max_context_sentence = len(sentence)

        # context_sentence
        for context_sentences in context_sentence_pad_batch:
            if len(context_sentences) < max_context:
                for i in range(max_context - len(context_sentences)):
                    context_sentences.append(['__eot__'])

        context_len_batch = []
        # context_sentence_len
        for context_sentence_len in context_sentence_len_pad_batch:
            context_len_batch.append(len(context_sentence_len))

            if len(context_sentence_len) < max_context:
                context_sentence_len.extend([0] * (max_context - len(context_sentence_len)))

        return np.array(context_sentence_pad_batch), np.array(context_sentence_len_pad_batch), \
               np.array(context_len_batch)

    def _get_speakers_id(self, speakers_batch):
        speaker_len = []
        # print("#"*200)
        for batch in speakers_batch:
            # print(batch)
            for i, speaker in enumerate(batch):
                batch[i] = self.speaker2id[speaker]
            speaker_len.append(len(batch))

        if max(speaker_len) > MAX_DIALOG_LENGTH:
            max_len = MAX_DIALOG_LENGTH
        else:
            max_len = max(speaker_len)

        #padding
        for i, batch in enumerate(speakers_batch):
            if len(batch) < max_len:
                batch.extend([-1]*(max_len - len(batch)))
            elif len(batch) > max_len:
                speakers_batch[i] = batch[-1 - max_len:-1]

        return speakers_batch

    def get_batch_data(self, batch_size, num_candidates=10):
        context_batch, context_len_batch = [], []
        utterances_batch, utterances_len_batch = [], []
        target_batch = []
        example_id_batch = []
        candidates_id_batch = []

        speaker_batch = []
        context_sentence_batch, context_sentence_len_batch = [], []

        data_check_index = 0
        for i in range(batch_size):

            data = next(self.dialog_iter,None)

            if data is None:
                break

            self.index += 1
            example_id, speaker, context, utterances, target_id, candidates_id = data

            example_id_batch.append(example_id)

            if num_candidates < 100:
                utterances, target_id, sampling_candidates_id = \
                    self.response_negative_sampling(num_candidates, utterances, target_id[0], candidates_id)
                candidates_id_batch.append(sampling_candidates_id)
            else:
                candidates_id_batch.append(candidates_id)

            # tokenized data(word)
            tokenized_context, tokenized_context_len = self.tokenize(context)
            tokenized_utterances, tokenized_utterances_len = self.tokenize(utterances)

            # tokenized data(index)
            context_batch.append(self.get_data_id(tokenized_context))
            utterances_batch.append(self.get_data_id(tokenized_utterances))


            # tokenized context sentences
            tok_context_s, tok_context_slen = self.get_context_sentence(context)
            context_sentence_batch.append(self.get_data_id(tok_context_s))
            context_sentence_len_batch.append(tok_context_slen)
            speaker_batch.append(speaker)

            # inputs length
            context_len_batch.extend(tokenized_context_len)
            utterances_len_batch.append(tokenized_utterances_len)

            target_batch.extend(target_id)

            data_check_index += 1

        #context sentence maximum value
        pad_context_sentence, context_sentence_len, tot_context_len = \
            self.get_context_sentence_padding(context_sentence_batch, context_sentence_len_batch)

        if data_check_index == 0:
            return None

        speaker_pad_batch = self._get_speakers_id(speaker_batch)

        max_context_len, max_utterance_len = 0, 0

        # get maximum input length
        max_context_len = max(context_len_batch)

        for each_len in utterances_len_batch:
            max_each_len = max(each_len)
            if max_utterance_len < max_each_len:
                max_utterance_len = max_each_len

        pad_context = self.padding_processing(context_batch, context_len_batch, max_context_len)
        pad_utterances = self.padding_processing(utterances_batch, utterances_len_batch, max_utterance_len)

        pad_context = np.asarray(pad_context).squeeze(1)
        pad_utterances = np.asarray(pad_utterances)
        speaker_pad_batch = np.asarray(speaker_pad_batch)

        return (pad_context, context_len_batch), \
               (pad_utterances, utterances_len_batch), target_batch, \
               (pad_context_sentence, context_sentence_len, tot_context_len), speaker_pad_batch, \
               example_id_batch, candidates_id_batch


    def get_batch_data_elmo(self, batch_size, num_candidates=10):

        context_batch, context_len_batch = [], []
        utterances_batch, utterances_len_batch = [], []
        target_batch = []

        speaker_batch = []
        context_sentence_batch, context_sentence_len_batch, tot_context_len = [], [], []
        example_id_batch = []
        candidates_id_batch = []

        data_check_index = 0
        for i in range(batch_size):

            data = next(self.dialog_iter, None)

            if data is None:
                break

            self.index += 1
            example_id, speaker, context, utterances, target_id, candidates_id = data
            example_id_batch.append(example_id)

            if num_candidates < 100:
                utterances, target_id, sampling_candidates_id = \
                    self.response_negative_sampling(num_candidates, utterances, target_id[0], candidates_id)
                candidates_id_batch.append(sampling_candidates_id)
            else:
                candidates_id_batch.append(candidates_id)

            # tokenized data(word)
            tokenized_context, tokenized_context_len = self.tokenize(context)
            tokenized_utterances, tokenized_utterances_len = self.tokenize(utterances)
            self.make_word_to_lower(tokenized_context)
            self.make_word_to_lower(tokenized_utterances)

            context_batch.append(tokenized_context)
            utterances_batch.append(tokenized_utterances)

            tok_context_s, tok_context_slen = self.get_context_sentence(context)
            self.make_word_to_lower(tok_context_s)
            context_sentence_batch.append(tok_context_s)
            context_sentence_len_batch.append(tok_context_slen)
            speaker_batch.append(speaker)

            # inputs length
            context_len_batch.extend(tokenized_context_len)
            utterances_len_batch.append(tokenized_utterances_len)

            target_batch.extend(target_id)

            data_check_index += 1

        pad_context_sentence, context_sentence_len, tot_context_len = \
            self.get_context_sentence_padding_elmo(context_sentence_batch, context_sentence_len_batch)

        if data_check_index == 0:
            return None

        speaker_pad_batch = self._get_speakers_id(speaker_batch)
        max_context_len, max_utterance_len = 0, 0

        for each_len in utterances_len_batch:
            max_each_len = max(each_len)
            if max_utterance_len < max_each_len:
                max_utterance_len = max_each_len

        context_batch = np.array(context_batch)
        utterances_batch = np.array(utterances_batch)
        speaker_pad_batch = np.array(speaker_pad_batch)

        return (context_batch, context_len_batch), (utterances_batch, utterances_len_batch), target_batch, \
               (pad_context_sentence, context_sentence_len, tot_context_len), speaker_pad_batch, \
               example_id_batch, candidates_id_batch

class AsyncDataSetWrapper(threading.Thread):
    def __init__(self, hparams, data_path, data_pickle, word2id, batch_size:int, num_candidates:int,
                 lock:threading.Lock, event:threading.Event, word_embedding_type="glove"):

        super().__init__()

        self._logger = logging.getLogger(__name__)

        self.hparams = hparams
        self.data_path = data_path
        self.word2id = word2id
        self.batch_size = batch_size
        self.num_candidates = num_candidates

        self.word_embedding_type = word_embedding_type

        self.batch_data_list = []
        if self.word_embedding_type == "elmo":
            self.batch_data_pickle_path = data_pickle % (self.batch_size, self.num_candidates, "_elmo")
        else:
            self.batch_data_pickle_path = data_pickle % (self.batch_size, self.num_candidates, "")

        if os.path.exists(self.batch_data_pickle_path):
            print("Pickled batch data is loading now")
            with open(self.batch_data_pickle_path, "rb") as f_handle:
                self.batch_data_list = pickle.load(f_handle)
            print("Pickled batch data loads - Complete! batch data length : %d" % len(self.batch_data_list))

        self.lock = lock
        self.event = event
        self._kill = False

        self.daemon = True

        self.start()
        # self._logger.debug("AsyncDataSetWrapper process started. (PID: %d)" % self.pid)

    def run(self):
        print("Thread has beed made")
        print("Word Embedding Data Type is %s" % self.word_embedding_type)
        while not self._kill:
            print("Thread is waiting for an event")
            self.event.wait()

            print("Thread Data Process!")
            batch_data_list = []
            data_process = DataProcess(self.hparams, self.data_path, "train", self.word2id)
            batch_index = 0
            self.batch_index = 0

            while not self._kill:
                if self.word_embedding_type == "glove":
                    batch_data = data_process.get_batch_data(self.batch_size, self.num_candidates)
                elif self.word_embedding_type == "elmo":
                    batch_data = data_process.get_batch_data_elmo(self.batch_size, self.num_candidates)

                batch_index += 1
                if batch_index % 1000 == 0:
                    print(batch_index, " data is loaded!")
                if batch_data is None:
                    self.batch_data_list = batch_data_list.copy()
                    print("data copy complete!")
                    print("total data length is %d" % len(batch_data_list))

                    if not os.path.exists(self.batch_data_pickle_path):
                        with open(self.batch_data_pickle_path, "wb") as f_handle:
                            pickle.dump(batch_data_list, f_handle)
                            print("pickle data save complete!")

                    print("Thread load train_batch_data_finished!")
                    break

                batch_data_list.append(batch_data)
                self.batch_index += 1

    def get_batch_data(self):
        return self.batch_data_list

    def kill(self):
        self._kill = True

    # def __getattr__(self, item):
    #     return getattr(self._data_set, item)









