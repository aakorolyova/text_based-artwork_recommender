import io
import string

import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk import word_tokenize

en_stop = stopwords.words('english')


def load_vec(emb_path, nmax=50000):
    """
    Loading embeddings from emb_path
    :param emb_path: path of embeddings file
    :param nmax: maximum number of word embeddings to load
    :return: embeddings vector, id2word dict, word2id dict
    """
    vectors = []
    word2id = {}
    print('Loading embeddings...')
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in tqdm(enumerate(f)):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


def get_word_embeddings(filepath, embedding_dim):
  word_embeddings = {}
  with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
      word, *vector = line.split()
      word_embeddings[word] = np.array(
          vector, dtype=np.float32)[:embedding_dim]

    return word_embeddings


def encode_phrase(phrase, emb, id2word):
    word2id = {v: k for k, v in id2word.items()}
    phrase = phrase.replace('/', ' / ').lower()   # nltk does not tokenise correctly is "/" not divided from words by spaces
    phrase_tok = word_tokenize(phrase)
    if len(phrase_tok) > 1:      # if length of the phrase is 1 word, filtering stop words is not performed
        phrase_tok = [word for word in phrase_tok if
                      word not in en_stop and word not in string.punctuation]
    phrase_emb = sum([emb[word2id[word.lower()]] for word in phrase_tok if word.lower() in word2id]) / len(phrase_tok)
    return phrase_emb


def encode_list_of_phrases(phrases, emb, id2word):
    vectors = []
    phrase2id = {}
    i = 0
    for phrase in phrases:
        phrase_emb = encode_phrase(phrase, emb, id2word)
        if isinstance(phrase_emb, float) and phrase_emb == 0.:
            continue
        vectors.append(phrase_emb)
        phrase2id[phrase.lower()] = i
        i += 1

    id2phrase = {v: k for k, v in phrase2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2phrase, phrase2id


def get_nn(word, src_emb, src_id2word, K=10):
    """
    Find K nearest neighbours in target language for a word in source language
    :param word: word for which nearest neighbours are retrieved
    :param src_emb: source language embeddings
    :param src_id2word: source language id2word dict
    :param tgt_emb: target language embeddings
    :param tgt_id2word: target language id2word dict
    :param K: number of nearest neighbours to return
    :return: a tuple of word and score for each neighbour
    """
    word2id = {v: k for k, v in src_id2word.items()}
    nneigbours = []
    if word.lower() in word2id:
        word_emb = src_emb[word2id[word.lower()]]
        scores = (src_emb / np.linalg.norm(src_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
        k_best = scores.argsort()[-K - 1:][
                 ::-1]  # for single language usage, top nearest word is the word itself, using K - 1 to get K different words
        for i, idx in enumerate(k_best):
            if src_id2word[idx] != word.lower():
                nneigbours.append((src_id2word[idx], scores[idx]))
    return nneigbours[:K]


if __name__=='__main__':
    embedding_dim = 100
    embeddings, id2word, word2id = load_vec(r'glove.6B.100d.txt', nmax=None)
    data = pd.read_csv('artworks.csv')

    artwork_name = data.artwork_name
    phrase_embeddings_name, id2phrase_name, phrase2id_name = encode_list_of_phrases(artwork_name, embeddings, id2word)

    name = 'Starry sky with yellow and clouds'
    print(get_nn(name, phrase_embeddings_name, id2phrase_name))


    artwork_desc = sorted(set(data[data.artwork_desc != ' '].artwork_desc))
    phrase_embeddings_desc, id2phrase_desc, phrase2id_desc = encode_list_of_phrases(artwork_desc, embeddings, id2word)

    desc = 'Juan Usl paintings are complex interactions and incorporate a great diversity of art historical references sensory and mental impressions various pictorial languages the gesture of painting and how the matter paints itself functions The first step is the canvas s preparation with multiple layers of gesso which is a key element that will remain visible This continuous manifestation of the gesso also conveys a philosophical resonance for Usl namely that the beginning is present at the end that the painting is a self contained entity a complete object not merely within its four sides but in the vertical layering of its surface as well The process of painting consists of a natural harmony between the manual act and the intellectual decisions Movement or even better displacement is a thematic key in his work In Usl s work we can read a constant dichotomy between opposing and complementary elements at the same time order and chaos presence and absence flatness and depth Most of his paintings are a juxtaposition of color areas and lines structures that seem to come and go like the fragments of a story The Artist presents his work as a temporary delimitation of infinite surfaces or as fragments from an infinite structure of lines Particularly well known is the series of paintings called So que Revelabas Dream that revelead In these works the artist through a deeply introspective practice seems to give shape to his more intimate self While painting Usl tries to connect rhythmically with his palpitation making each stroke a symbolic representation of the beating of his heart return the theme of disorientation no longer understood only in a physical sense but also and above all in a more temporal and perceptive way His paintings take the viewer into a labyrinthine space in which the articulation seems to indicate a specific direction while paradoxically it leaves open the way to interpretation '
    print(get_nn(desc, phrase_embeddings_desc, id2phrase_desc))
