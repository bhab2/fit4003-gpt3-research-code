import re
from sklearn.cluster import KMeans
from cmath import isnan
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import webcolors
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from gensim.parsing.preprocessing import remove_stopwords
import plotly.express as px
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

# Libraries for preprocessing
# Download once if using NLTK for preprocessing
nltk.download('punkt')

# Libraries for vectorisation
# from fuzzywuzzy import fuzz
# Libraries for clustering
# Load data set
df = pd.read_csv('')
# text1 = df['Summary']
df.head()
text1 = df['Review Summary']

for i in range(len(text1)):
    if text1[i] == ' ':
        text1.pop(i)
        continue
    elif type(text1[i]) == float:
        if isnan(text1[i]):
            text1.pop(i)
            continue
        else:
            text1[i] = str(text1[i])
            continue
    else:
        if 'jasper' in text1[i].lower():
            text1[i] = text1[i].lower().replace('jasper', '')
        if 'jarvis' in text1[i].lower():
            text1[i] = text1[i].lower().replace('jarvis', '')
        if 'messagebird' in text1[i].lower():
            text1[i] = text1[i].lower().replace('messagebird', '')
        if 'cheatlayer' in text1[i].lower():
            text1[i] = text1[i].lower().replace('cheatlayer', '')

text1 = text1.reset_index(drop=True)

# Remove stopwords, punctuation and numbers
text2 = [remove_stopwords(x)
         .translate(str.maketrans('', '', string.punctuation))
         .translate(str.maketrans('', '', string.digits))
         for x in text1]


# Stem and make lower case
def stemSentence(sentence):
    porter = PorterStemmer()
    token_words = word_tokenize(sentence)
    stem_sentence = [porter.stem(word) for word in token_words]
    return ' '.join(stem_sentence)


text3 = pd.Series([stemSentence(x) for x in text2])


# Remove colours
colors = list(webcolors.CSS3_NAMES_TO_HEX)
colors = [stemSentence(x) for x in colors if x not in ('bisque', 'blanchedalmond', 'chocolate', 'honeydew', 'lime',
                                                       'olive', 'orange', 'plum', 'salmon', 'tomato', 'wheat')]
text4 = [' '.join([x for x in string.split() if x not in colors])
         for string in text3]

# Bag of words
vectorizer_cv = CountVectorizer(analyzer='word')
X_cv = vectorizer_cv.fit_transform(text4)


# TF-IDF (word level)
vectorizer_wtf = TfidfVectorizer(analyzer='word')
X_wtf = vectorizer_wtf.fit_transform(text4)


# TF-IDF (n-gram level)
vectorizer_ntf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
X_ntf = vectorizer_ntf.fit_transform(text4)


# LDA
lda = LatentDirichletAllocation(n_components=10, learning_decay=1)
X_lda = lda.fit(X_cv)


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(5, 5, figsize=(30, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


# Show topics
n_top_words = 10  # Possibly show more than 5 words
feature_names = vectorizer_cv.get_feature_names()
plot_top_words(X_lda, feature_names, n_top_words, '')
