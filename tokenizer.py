import spacy
from sklearn.feature_extraction.text import CountVectorizer
import multiprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from joblib import dump, load
import pickle
import pandas as pd
import time
import gc
import numpy as np
import unidecode
import string
import re
from tqdm import tqdm
import math
import util
from parameters import DATA_PATH, language_dic, validation, file_posfix, padding_len
from datetime import date
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from embedding import getCorpus
import os

# defines a custom vectorizer class
class SimpleVectorizer(CountVectorizer):

    def build_analyzer(self):
        
        # create the analyzer that will be returned by this method
        def analyser(doc):
            
            tokens = str(doc).split(" ")
            
            # use CountVectorizer's _word_ngrams built in method
            # to remove stop words and extract n-grams
            return(self._word_ngrams(tokens, []))
        return(analyser)

def tokenizer(train_file, test_file, output_path=DATA_PATH+'embeddings/keras_tokenized_tri/'):
    print('Reading Files')
    df_train = pd.read_csv(train_file)
    df_train['title'] = df_train['title'].astype(str)
    df_test = pd.read_csv(test_file)
    df_test['title'] = df_test['title'].astype(str)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
            
    for lang in ['pt', 'es']:
        print('Generating tokenizer for '+ lang)
        df_train_lang_index = df_train[df_train['language'] == language_dic[lang]].index
        df_test_lang_index = df_test[df_test['language'] == language_dic[lang]].index
        corpus_lang = np.concatenate((df_train['title'].iloc[df_train_lang_index].values, df_test['title'].iloc[df_test_lang_index].values))
        
        #CountVectorizer to count the number of words with frequency higher than 5
        print('Counting word frequency')
        vec = SimpleVectorizer(min_df=5)
        vec_trans = vec.fit_transform(corpus_lang)
        vocab_size = len(vec.vocabulary_)
        del vec, vec_trans
        gc.collect()
        
        print('Generating tokens on '+ output_path)
        tokenizer = Tokenizer(num_words=vocab_size, lower=False, filters='')
        tokenizer.fit_on_texts(corpus_lang)
        dump(tokenizer, output_path+'tokenizer_'+lang+'.dump')
        
        tokenized_lang = tokenizer.texts_to_sequences(corpus_lang)
        dump(tokenized_lang[:df_train_lang_index.shape[0]], output_path+lang+'.dump')
        dump(tokenized_lang[df_train_lang_index.shape[0]:], output_path+lang+'_test.dump')
        del tokenizer, tokenized_lang, corpus_lang
        gc.collect()

if __name__ == '__main__':
    tokenizer(train_file=DATA_PATH+'train_preprocessed.csv', test_file=DATA_PATH+'test_preprocessed.csv', output_path=DATA_PATH+'embeddings/keras_tokenized_tri/')