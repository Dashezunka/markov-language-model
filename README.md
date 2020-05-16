# Text language classifier
A model of text language classifier based on the Markov chain. Works with Russian, Ukrainian and Belorussian languages.

## Description
The algorithm works in the following way:
- Prepares datasets for languages: tokenizes and lemmatizes corpuses. 
- Converts sentences to make a list of n-grams for every language.
- Divides datasets into train sets (for every language) and global test set.
- Fits a model on training datasets using Markov chain.
- Predicts a language to the test set samples.
- Evaluates the model using precision, recall and f1-metrics. 
- Saves results of the model evaluation in csv-files.

**Note:** 
For transition matrix storage of the Markov chain graph an adjacency list is used.

## Install and configure
You need to install dependencies from `requirements.txt` using
`pip3 install -r requirements.txt`  

**Note:** You can adjust *NGRAM_SIZE* and *MIN_PROBABILITY* in `options.py` before starting the model.  
You can also set your own *CORPUS_PATH* for every language.
## Running command
Try `python3 lang_classifier.py` in the project directory.