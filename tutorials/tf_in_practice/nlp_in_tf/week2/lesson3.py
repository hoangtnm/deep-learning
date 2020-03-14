import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


vocab_size = 10000
embedding_size = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for sentence, label in train_data:
    training_sentences.append(str(sentence.numpy()))
    training_labels.append(label.numpy())

for sentence, label in test_data:
    testing_sentences.append(str(sentence.numpy()))
    testing_labels.append(label.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

tokenizer = info.features['text'].encoder
