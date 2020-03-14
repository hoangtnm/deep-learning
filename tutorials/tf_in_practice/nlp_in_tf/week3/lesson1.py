import numpy as np
import tensorflow.keras as keras
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"
BUFFER_SIZE = 10000
BATCH_SIZE = 64
num_epochs = 50


# Load IMDB dataset
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Data pre-processing
training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)


# Model definition
model = keras.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, 64),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequence=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam,
              metrics=['accuracy'])


history = model.fit(padded, training_labels_final, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels_final))
