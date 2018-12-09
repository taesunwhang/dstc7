'''
ELMo usage example with pre-computed and cached context independent
token representations

Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from bilm import dump_token_embeddings

data_type = "advising"

print("dump %s token embeddings for ELMo" % data_type)
# Location of pretrained LM.  Here we use the test fixtures.
vocab_file = os.path.join('data', data_type, '%s_subtask_1.vocab.txt' % data_type)
options_file = os.path.join('pretrained', 'options', 'options_small.json')
weight_file = os.path.join('pretrained', 'weights', 'elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

# Dump the token embeddings to a file. Run this once for your dataset.
token_embedding_file = os.path.join('pretrained', 'token_embeddings', 'elmo_%s_1_relevant_token_embeddings.hdf5'
                                    % data_type)

dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)
print(data_type + "1 relevant Token dump finished")
