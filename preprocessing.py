import spacy
from multiprocessing import Pool
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
from parameters import DATA_PATH, validation, language_dic
import os
import gensim

extra_stop = ['original', 'barato', 'cuotas', 'pronta', 'frete', 'oferta',
       'promocion', 'promocao', 'envio', 'oportunidad', 'entrega',
       'produto', 'gratuito', 'producto', 'producao', 'interes',
       'produccion', 'promo', 'free', 'gratis']

# load spaCy's model for english language
model = {'pt': 'pt_core_news_sm', 'es': 'es_core_news_sm'}
#spacy.prefer_gpu()
nlp_pt = spacy.load(model['pt'], disable=['tagger', 'parser', 'ner'])
nlp_es = spacy.load(model['es'], disable=['tagger', 'parser', 'ner'])
stop_words_pt = list(nlp_pt.Defaults.stop_words)
stop_words_pt.extend(extra_stop)
stop_words_es = list(nlp_es.Defaults.stop_words)
stop_words_es.extend(extra_stop)
stop_words = {
    'pt': stop_words_pt,
    'es': stop_words_es
}

# Regex
special_caracters_list = re.escape(',`\'"|#=_![](){}<>^\\+/*?%.~:@;')
dash_word = re.compile(r' -(?=[^\W\d_])')
special_caracters = re.compile(r'[%s]' % (special_caracters_list))
normalize_numbers = re.compile(r'\d')
multiple_white_spaces = re.compile(r'\s\s+')

def normalize_doc(doc):
    doc = str(doc)
    doc = doc.lower()
    doc = dash_word.sub(' - ', doc)
    doc = special_caracters.sub(' ', doc)
    doc = normalize_numbers.sub('0', doc)
    doc = multiple_white_spaces.sub(' ', doc)
    return doc.strip()

def process_pt(doc):
    doc = normalize_doc(doc)
    doc = [unidecode.unidecode(token) for token in doc.split() if token not in stop_words['pt'] and len(token)>=2]
    return doc

def process_es(doc):
    doc = normalize_doc(doc)
    doc = [unidecode.unidecode(token) for token in doc.split() if token not in stop_words['es'] and len(token)>=2]
    return doc

process_func = {'es': process_es, 'pt':process_pt}

#process all at once in parallel
def preprocess_parallel(train_file='train.csv', test_file='train.csv', path=DATA_PATH, workers=4):
    start_time = time.time()
    train_target = train_file[:-4]+'_preprocessed.csv'
    test_target = test_file[:-4]+'_preprocessed.csv'
    print('Reading file...')
    df_train = pd.read_csv(path+train_file)
    df_test = pd.read_csv(path+test_file)
        
    for lang in ['pt', 'es']:
        print("Start processing for "+lang)
        df_train_lang_index = df_train[df_train['language'] == language_dic[lang]].index
        df_test_lang_index = df_test[df_test['language'] == language_dic[lang]].index

        #Save categories indices for easy loading later during model training
        np.save(path+'categories_int_'+lang+'.npy', util.int_encode(df_train['category'].iloc[df_train_lang_index].values))
        np.save(path+'test_index_'+lang+'.npy', df_test_lang_index)

        corpus_lang = np.concatenate((df_train['title'].iloc[df_train_lang_index].values, df_test['title'].iloc[df_test_lang_index].values))
        p = Pool(workers)
        corpus_lang_preprocessed = p.map(process_func[lang], corpus_lang)
        p.close()
        p.join()
        print(corpus_lang_preprocessed[0:5])
        del corpus_lang
        gc.collect()
        
        print("Traning trigram and bigram models")
        bigram =  gensim.models.phrases.Phrases(corpus_lang_preprocessed, min_count=100, threshold=20.0, max_vocab_size=10000000*16) # train model
        trigram =  gensim.models.phrases.Phrases(bigram[corpus_lang_preprocessed], min_count=500, threshold=20.0, max_vocab_size=10000000*16)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        dump(bigram_mod, path+'bigram_mod_'+lang+'.dump')
        dump(trigram_mod, path+'trigram_mod_'+lang+'.dump')
        
        print("Generating bigram/trigram on corpus")
        for i, doc in enumerate(tqdm(corpus_lang_preprocessed)):
            corpus_lang_preprocessed[i] = " ".join(trigram_mod[bigram_mod[doc]])
        print(corpus_lang_preprocessed[0:5])
        
        df_train['title'].iloc[df_train_lang_index] = corpus_lang_preprocessed[:df_train_lang_index.shape[0]]
        df_test['title'].iloc[df_test_lang_index] = corpus_lang_preprocessed[df_train_lang_index.shape[0]:]
        del corpus_lang_preprocessed, bigram, trigram, bigram_mod, trigram_mod, df_train_lang_index, df_test_lang_index
        gc.collect()
        
    print("Saving on disk")
    if os.path.exists(path+train_target):
        os.remove(path+train_target)
    if os.path.exists(path+test_target):
        os.remove(path+test_target)
    df_train.to_csv(path+train_target, index=False)
    df_test.to_csv(path+test_target, index=False)
        
    print("Preprocess time: --- %s seconds ---" % (time.time() - start_time))

def preprocess_single_text(text, lang='pt', path=DATA_PATH):
    bigram_mod = load(path+'bigram_mod_'+lang+'.dump')
    trigram_mod = load(path+'trigram_mod_'+lang+'.dump')
    text = process_func[lang](text)
    text = " ".join(trigram_mod[bigram_mod[text]])
    return text
    
#Separate some validation examples to compare models
def separate_val_files(file='train.csv', path=DATA_PATH, frac=0.2):
    df = pd.read_csv(path+file)
    for lang in ['es', 'pt']:
        df_lang =  df[df['language'] == language_dic[lang]]
        df_lang['lang_id'] = np.arange(len(df_lang))
        val_lang = df_lang[df_lang['label_quality'] == 'reliable'].sample(frac=0.2)
        np.save(path+'val_index_'+lang+'.npy', val_lang['lang_id'].values)
        
if __name__ == '__main__':
    preprocess_parallel(train_file='train.csv', test_file='test.csv', path=DATA_PATH, workers=4)
    separate_val_files(file='train_preprocessed.csv', path=DATA_PATH, frac=0.2)
    print('Intel core I7-8500h Novo PROMOÇÃO -> ', preprocess_single_text('Intel core I7-8500h Novo PROMOÇÃO', lang='pt'))