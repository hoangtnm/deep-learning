# Natural Language Processing in TensorFlow

This course is part of the [TensorFlow in Practice Specialization](https://www.coursera.org/specializations/tensorflow-in-practice)


## About this Course

In this course, you will build natural language processing systems using TensorFlow. You will learn to process text, including tokenizing and representing sentences as vectors, so that they can be input to a neural network. You’ll also learn to apply RNNs, GRUs, and LSTMs in TensorFlow. Finally, you’ll get to train an  LSTM on existing text to create original poetry!

## Syllabus - What you will learn from this course

### Week 1: Sentiment in Text

The first step in understanding sentiment in text, and in particular when training a neural network to do so is
the tokenization of that text. This is the process of converting the text into numeric values, with a number representing
a word or a character. This week you'll learn about the `Tokenizer` and `pad_sequences` APIs in TensorFlow and how they
can be used to prepare and encode text and sentences to get them ready for training neural networks!

<!-- - Lesson 1: Word based encodings
- Lesson 2: Text to sequence -->

### Week 2: Word Embeddings

Last week you saw how to use the Tokenizer to prepare your text to be used by a neural network by converting words into
numeric tokens, and sequencing sentences from these tokens. This week you'll learn about Embeddings, where these tokens
are mapped as vectors in a high dimension space. With Embeddings and labelled examples, these vectors can then be tuned
so that words with similar meaning will have a similar direction in the vector space. This will begin the process of
training a neural network to udnerstand sentiment in text -- and you'll begin by looking at movie reviews, training a
neural network on texts that are labelled 'positive' or 'negative' and determining which words in a sentence drive those 
meanings.

### Week 3: Sequence Models

### Week 4: Sequence Models and Literature