from os.path import join, isfile
from os import listdir
import csv
from sklearn.model_selection import train_test_split

from text_preprocessing import *
from markov_language_model import *


class Metrics:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def print(self):
        print("TP={0}\tFP={1}\nFN={2}\tTN={3}".format(self.tp, self.fp, self.fn, self.tn))

    def accuracy(self):
        return self.tp + self.tn / (self.tp + self.fp + self.fn + self.tn)

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return 2 * precision * recall / (precision + recall)


class ModelWrapper:
    def __init__(self, language):
        self.model = None
        self.language = language
        self.test_set = []
        self.metrics = Metrics()


def read_language_corpus(path):
    corpus = []
    document_name_list = [f for f in listdir(path) if isfile(join(path, f))]
    for name in document_name_list:
        with open(join(path, name), 'r') as doc:
            text = doc.read()
            document_sentences = text_preprocessing(text)
            corpus.extend(document_sentences)
    return corpus


# [  0   ,   1   ,   2   ]
# ['Мама', 'мыла', 'раму']
# [ ['__start_token__', 'Мама'], ['мама', 'мыла'], ['мыла', 'раму'] ]


def convert_sentence_to_ngrams(sentence):
    result = []
    for index, token in enumerate(sentence):
        if index < NGRAM_SIZE - 1:
            ngram = ['__start_token__'] * (NGRAM_SIZE - 1 - index) + sentence[:index + 1]
            result.append(ngram)
        else:
            ngram = sentence[index - (NGRAM_SIZE - 1):index + 1]
            result.append(ngram)
    return result


russian_corpus = read_language_corpus(RUS_CORPUS_PATH)
ukraine_corpus = read_language_corpus(UKR_CORPUS_PATH)
belarus_corpus = read_language_corpus(BYR_CORPUS_PATH)

corpus_dict = {
    'ru': russian_corpus,
    'ukr': ukraine_corpus,
    "byr": belarus_corpus
}
lang_ngrams_dict = {}

for language, corpus in corpus_dict.items():
    print("Corpus processing for ", language)
    lang_ngrams_dict[language] = []
    for sentence in corpus:
        sentence_ngrams = convert_sentence_to_ngrams(sentence)
        lang_ngrams_dict[language].append(sentence_ngrams)

models = []

for language, ngrams in lang_ngrams_dict.items():
    print("For language {0} have {1} sentences. Sum of Words in corpus {2}".format(language, len(ngrams), sum(
        [len(s_ngrams) for s_ngrams in ngrams])))
    X_train, X_test = train_test_split(ngrams, test_size=0.2, random_state=0)
    model_wrap = ModelWrapper(language)
    model_wrap.test_set = X_test
    language_model = MarkovChainLanguageModel(language)
    model_train_set = []
    for sentence_ngrams in X_train:
        model_train_set.extend(sentence_ngrams)
    language_model.fit(model_train_set)
    model_wrap.model = language_model
    models.append(model_wrap)

full_test_set = []
for model in models:
    lang = model.language
    for sentence_ngrams in model.test_set:
        full_test_set.append((lang, sentence_ngrams))

true_counter = 0
error_counter = 0
for true_label, test_sentence in full_test_set:
    language_probability_list = [(m.model.predict(test_sentence), m.language) for m in models]
    max_proba, predicted_label = max(language_probability_list)
    if true_label == predicted_label:
        true_counter += 1
        for m in models:
            if m.language != true_label:
                m.metrics.tn += 1
            else:
                m.metrics.tp += 1
    else:
        error_counter += 1
        for m in models:
            if m.language == true_label:
                m.metrics.fn += 1
            if m.language == predicted_label:
                m.metrics.fp += 1

with open('data/results/{0}gram_report.csv'.format(NGRAM_SIZE), 'w') as out:
    out_writer = csv.writer(out)
    out_writer.writerow(['Language', 'Precision', 'Recall', 'F1-score'])
    for model in models:
        precision = model.metrics.precision()
        recall = model.metrics.recall()
        f1_score = model.metrics.f1()
        print("Metrics for {0} language:".format(model.language))
        print("\tPrecision={:.3f}".format(precision))
        print("\tRecall={:.3f}".format(recall))
        print("\tF1={:.3f}".format(f1_score))
        out_writer.writerow([model.language, precision, recall, f1_score])

