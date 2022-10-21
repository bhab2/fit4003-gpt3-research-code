from gensim.models import TfidfModel
import warnings
import pyLDAvis.gensim_models
import pyLDAvis
import numpy as np
import glob
import pandas as pd
from cmath import isnan

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess


import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('')
df.head()

dataArray = []

for item in df['Review Summary']:
    if item != ' ':
        if type(item) == float:
            if isnan(item):
                continue
            else:
                dataArray.sppend(str(item))
                continue

        dataArray.append(item.strip())

stopwords = stopwords.words("english")


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []

    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)

        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)


lemmatized_texts = lemmatization(dataArray)


def gen_words(texts):
    final = []

    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)

    return (final)


data_words = gen_words(lemmatized_texts)

# BIGRAMS AND TRIGRAMS
bigrams_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=30)
trigrams_phrases = gensim.models.Phrases(
    bigrams_phrases[data_words], threshold=30)

bigram = gensim.models.phrases.Phraser(bigrams_phrases)
trigram = gensim.models.phrases.Phraser(trigrams_phrases)


def make_bigrams(texts):
    return (bigram[doc] for doc in texts)


def make_trigram(texts):
    return (trigram[doc] for doc in texts)


data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = list(make_trigram(data_bigrams))


# TF-IDF REMOVAL

id2word = corpora.Dictionary(data_bigrams_trigrams)

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus, id2word=id2word)
low_value = 0.03
words = []
words_missing_in_tfidf = []

for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = []
    tfidf_ids = [id for id, value in bow]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words + words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]

    new_bow = [b for b in bow if b[0]
               not in low_value_words and b[0] not in words_missing_in_tfidf]
    corpus[i] = new_bow

id2word = corpora.Dictionary(data_words)

print(id2word)

corpus = []

for text in data_words:
    new = id2word.doc2bow(text)
    corpus.append(new)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=10,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha="auto")


pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim_models.prepare(
    lda_model, corpus, id2word, mds="mmds", R=30, lambda_step=0.02, sort_topics=True)
