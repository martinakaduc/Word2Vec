from underthesea import sent_tokenize, word_tokenize
from file_utils import *

def load_corpus(corpus_file='truyen_kieu.txt', dictionary='dictionary.txt'):
    """
    :param nb_sentences: Use if all brown sentences are too many
    :return: index2word (list of string)
    """
    corpus = load_data(corpus_file)
    print ('Building vocab ...')

    corpus = sent_tokenize(corpus)
    for i, sentence in enumerate(corpus):
        corpus[i] = word_tokenize(sentence)

    vocab = list(set([word.replace(' ', '_') for sent in corpus for word in sent]))
    with open(dictionary, 'w', encoding='utf8') as f:
        f.write('\n'.join(vocab))

    # ids: list of (list of word-id)
    ids = [[vocab.index(w) for w in sent if w in vocab]
           for sent in corpus]

    return ids, vocab


def skip_grams(sentences, window, vocab_size, nb_negative_samples=5.):
    """
    calc `keras.preprocessing.sequence.skipgrams` for each sentence
    and concatenate those.

    :param sentences: list of (list of word-id)
    :return: concatenated skip-grams
    """
    import keras.preprocessing.sequence as seq
    import numpy as np

    print ('Building skip-grams ...')

    def sg(sentence):
        return seq.skipgrams(sentence, vocab_size,
                             window_size=np.random.randint(window - 1) + 1,
                             negative_samples=nb_negative_samples)

    couples = []
    labels = []

    # concat all skipgrams
    for cpl, lbl in map(sg, sentences):
        couples.extend(cpl)
        labels.extend(lbl)

    return np.asarray(couples), np.asarray(labels)


def save_weights(model, index2word, vec_dim, filename = 'word_vector.txt'):
    """
    :param model: keras model
    :param index2word: list of string
    :param vec_dim: dim of embedding vector
    :return:
    """
    vec = model.get_weights()[0]
    f = open(filename, 'w', encoding='utf8')
    # first row in this file is vector information
    f.write(" ".join([str(len(index2word)), str(vec_dim)]))
    f.write("\n")

    for i, word in enumerate(index2word):
        f.write(word)
        f.write(" ")
        f.write(" ".join(map(str, list(vec[i, :]))))
        f.write("\n")
    f.close()


def most_similar(positive=[], negative=[], filename = 'word_vector.txt'):
    """
    :param positive: list of string
    :param negative: list of string
    :return:
    """
    from gensim import models
    vec = models.KeyedVectors.load_word2vec_format(filename, binary=False, encoding='utf8')
    for v in vec.most_similar_cosmul(positive=positive, negative=negative, topn=20):
        print(v)
