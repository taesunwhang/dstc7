import numpy as np
import pickle
from data_process import DataProcess
import nltk

def get_vocabs(filename, type="train"):
    data_process = DataProcess(filename, type)
    dialog_iter = data_process.dialog_iter

    vocab = set()
    index = 0
    while True:
        # data -> (context, utterances, target_id)
        data = next(dialog_iter, None)

        if data is None:
            break

        example_id, speaker, context, utterances, target_id, candidates_id = data

        tokenized_context, _ = data_process.tokenize(context)
        tokenized_utterances, _ = data_process.tokenize(utterances)

        for context_sentence in tokenized_context:
            for i, c_word in enumerate(context_sentence):
                context_sentence[i] = c_word.lower()
            vocab.update(context_sentence)

        for utterances_sentence in tokenized_utterances:
            for i, u_word in enumerate(utterances_sentence):
                utterances_sentence[i] = u_word.lower()
            vocab.update(utterances_sentence)

        index += 1
        if index % 100 == 0:
            print(index, len(vocab))

    return vocab

def save_pickle_data(filename, pickle_filename):
    data_process = DataProcess(filename)
    dialog_iter = data_process.create_dialogue_iter(data_process.input_file_path)

    index = 0
    with open(pickle_filename, 'wb') as f_handle:
        while True:
            # data -> (context, utterances, target_id)
            data = next(dialog_iter, None)

            if data is None:
                break

            context, utterances, target_id = data

            tokenized_context, _ = data_process.tokenize(context)
            tokenized_utterances, _ = data_process.tokenize(utterances)

            save_data = [tokenized_context,tokenized_utterances]
            pickle.dump(save_data, f_handle)

            index += 1

            if index % 100 == 0:
                print(index)

    print("%s data save complete!" % index)

def write_char_vocab(vocab,filename):
    char_vocab = set()
    train_vocab = vocab[1:-1]
    for word in vocab:
        char_vocab.update(list(word))

    with open(filename,"w", encoding="utf-8") as f_handle:
        for i, char in enumerate(char_vocab):
            if i == len(char_vocab) - 1:
                f_handle.write(char)
            else:
                f_handle.write("%s\n" % char)

def load_char_vocab(filename):
    char_vocab = None

    with open(filename, "r") as f_handle:
        char_vocab = f_handle.read().splitlines()

    char2id = dict()
    char_vocab.insert(0, "<PAD>")
    char_vocab.append("<UNK>")

    for idx, char in enumerate(char_vocab):
        char2id[char] = idx

    return [char_vocab, char2id]

def write_vocab(vocab, filename):
    """
        Writes a vocab to a file
        Args:
            vocab: iterable that yields word
            filename: path to vocab file
        Returns:
            write a word per line
        """
    with open(filename,"w", encoding="utf-8") as f_handle:
        for i, word in enumerate(vocab):
            if i == len(vocab) - 1:
                f_handle.write(word)
            else:
                f_handle.write("%s\n" % word)

    print("Write Vocabulary is done. %d tokens" % len(vocab))

def load_vocab(filename):
    vocab = None
    with open(filename) as f_handle:
        vocab = f_handle.read().splitlines()

    word2id = dict()
    vocab.insert(0,"<PAD>")
    vocab.append("<UNK>")

    for idx, word in enumerate(vocab):
        word2id[word] = idx

    return [vocab, word2id]



def load_glove_vocab(filename):
    """
    Args:
        filename: path to the glove vectors hparams.glove_path
    """
    print("Building glove vocab...")
    vocab = set()
    with open(filename, "r", encoding='utf-8') as f_handle:
        for line in f_handle:
            word = line.strip().split(' ')[0]
            # print(word)
            vocab.add(word)

    print("Getting Glove Vocabulary is done. %d tokens" % len(vocab))

    return vocab

def export_trimmed_glove_vectors(word2id, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array
    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.random.uniform(low=-1,high=1,size=(len(word2id), dim))
    print(embeddings.shape)

    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]

            if len(embedding) < 2:
                continue

            if word in word2id:
                word_idx = word2id[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def make_valid_data(filename, write_file_name):

    data_process = DataProcess(filename)
    dialog_iter = data_process.create_dialogue_iter(data_process.input_file_path)

    input_sum_turn = 0
    input_sum_sentence_len = 0
    with open(write_file_name, "w", encoding='utf-8') as f_handle:
        index = 0

        while True:
            index += 1
            # data -> (context, utterances, target_id)
            data = next(dialog_iter, None)

            if data is None:
                break
            speakers, context, utterances, target_id = data
            context_sentence = context[0].split(" __eot__ ")

            f_handle.write("[%d]" % index + "\n")
            sum_sentence_len = 0
            tot_turn = 0

            for i, sentence in enumerate(context_sentence):
                sentence_len = len(nltk.word_tokenize(sentence))
                if len(sentence) == 0:
                    continue
                sentence_string = speakers[i] + " : " + sentence
                sentence_string = str(sentence_len) + "|" + sentence_string
                f_handle.write(sentence_string + "\n")

                sum_sentence_len += sentence_len
                tot_turn += 1

            avg_sentence_len = sum_sentence_len / tot_turn
            sentence_answer = "Answer : " + utterances[target_id[0]] + "\n"
            f_handle.write(sentence_answer)
            f_handle.write("average sentence length : %.3f" % avg_sentence_len + "\n")
            f_handle.write("total turn number : %d" % tot_turn + '\n')

            f_handle.write("-"*200 + "\n")
            if index % 500 == 0:
                print(index, ":", "avg_sentence_len - %.3f" % avg_sentence_len, "tot_turn - %d" % tot_turn)
            input_sum_turn += tot_turn
            input_sum_sentence_len += avg_sentence_len

        f_handle.write("average sentence length %.3f" % (input_sum_sentence_len / index))
        f_handle.write("average turn length %.3f" % (input_sum_turn / index))

def load_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data["embeddings"]

def data_stat(file_type:str, filename):
    """

    :param filename:
    :return:
    """
    data_process = DataProcess(filename, "test")
    dialog_iter = data_process.dialog_iter

    index = 0
    #tokenized context len(word)
    tot_context_len = 0
    max_context_len = 0
    min_context_len = 10000

    #context # of turn
    tot_num_context_turn = 0
    max_context_turn = 0
    min_context_turn = 10000

    with open("./stat/%s.txt" % file_type, "w") as f_handle:

        while True:
            # data -> (context, utterances, target_id)
            data = next(dialog_iter, None)

            if data is None:
                f_handle.write("##################################################################\n\n")
                f_handle.write("%s\n" % filename)

                f_handle.write("Average # of each context turn : %.3f\n" % (tot_num_context_turn / index))
                f_handle.write("Maximum # of context turn : %d\n" % max_context_turn)
                f_handle.write("Minimum # of context turn : %d\n\n" % min_context_turn)

                f_handle.write("Average of Context Sentence Length : %.3f\n" % (tot_context_len / index))
                f_handle.write("Maximum len context : %d\n" % max_context_len)
                f_handle.write("Minimum len context : %d\n" % min_context_len)

                exit()

            example_id, speaker, context, utterances, target_id, candidates_id = data

            num_context_turn= len([c for c in context[0].strip().split("__eou__  __eot__") if len(c) > 0])

            #each tokenized context length
            tokenized_context, context_len = data_process.tokenize(context)

            tot_context_len += context_len[0]

            #maximum_context_len
            if max_context_len < context_len[0]:
                max_context_len = context_len[0]

            # minimum_context_len
            if min_context_len > context_len[0]:
                min_context_len = context_len[0]

            #number of context turn(how many turns exist)
            tot_num_context_turn += num_context_turn
            #maximum turn(number)
            if max_context_turn < num_context_turn:
                max_context_turn = num_context_turn

            #minimum_turn(number)
            if min_context_turn > num_context_turn:
                min_context_turn = num_context_turn

            index += 1
            if index % 1000 == 0:
                print(index, ":", "%.3f" % (tot_num_context_turn / index))