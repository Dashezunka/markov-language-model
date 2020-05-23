import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet, stopwords
from pymystem3 import Mystem

m = Mystem()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def tokenize_text(text):
    result = []
    for sentence in sent_tokenize(text):
        sentence = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "__number_token__", sentence)
        sentence = re.sub(r'[^\w\s]', '', sentence)
        tokens = word_tokenize(sentence.lower())
        result.append(tokens)
    return result


def lemmatize_text(tokens):
    lemmas = list()
    for token in tokens:
        if token == '__number_token__':
            lemmas.append(token)
        lemmas.append(m.lemmatize(token)[0])
    return lemmas


def remove_stop_words(tokens):
    stopwords_set = set(stopwords.words('russian'))
    return [token for token in tokens if token not in stopwords_set]


def text_preprocessing(text):
    result = []
    tokenized_sentences = tokenize_text(text)
    for sentence in tokenized_sentences:
        tokens = lemmatize_text(sentence)
        #tokens = remove_stop_words(tokens)
        result.append(tokens)
    return result
