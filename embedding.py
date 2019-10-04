from parameters import DATA_PATH, file_posfix, language_dic
import pandas as pd
import numpy as np
from joblib import load, dump
import gensim
import gc
import time
from keras.preprocessing.text import Tokenizer
import unidecode
from nltk.stem.snowball import SnowballStemmer
#from spellchecker import SpellChecker
import spacy
import tqdm
import os

def csv_reader_generator(file):
    for df in pd.read_csv(file, chunksize=10000, usecols=["title"], dtype={'title': str}):
        for doc in df['title'].values:
            yield str(doc).split()
            
def getCorpus(train_file, test_file):
    train = pd.read_csv(train_file, usecols = ['title', 'language'])
    test = pd.read_csv(test_file, usecols = ['title', 'language'])
    train = pd.concat([train, test], ignore_index=True)
    train['title'] = train['title'].apply(str)
    train_pt = train[train['language'] == 'portuguese'].copy()
    train_es = train[train['language'] == 'spanish'].copy()
    
    corpus_pt, corpus_es = train_pt['title'].values, train_es['title'].values
    
    del train, test
    del train_pt, train_es
    gc.collect()
    return corpus_pt, corpus_es
    """
    f_pt = open('./data/corpus_pt.txt', 'r').read().splitlines()
    f_es = open('./data/corpus_es.txt', 'r').read().splitlines()
    for f in [f_pt, f_es]:
        for i, text in enumerate(f):
            f[i] = text.split()
    print(f_pt[1:10])
    print(f_es[1:10])
    return f_pt, f_es
    """
    
def generateCorpus(train_file, test_file, output_path=DATA_PATH):
    train_pt, train_es = getCorpus(train_file, test_file)
    
    for train_lang, lang in [(train_pt, 'pt'), (train_es, 'es')]:
        path = output_path+'corpus_'+lang+'.txt'
        if not os.path.exists(path):
            os.remove(path)
        with open(path, 'w') as file:
            for line in train_lang['title'].values:
                file.write(str(line)+'\n')
    
    return [output_path+'corpus_pt.txt', output_path+'corpus_es.txt']
        
def fasttext(train_file, test_file, path=DATA_PATH, workers=4):
    train_pt, train_es = getCorpus(train_file, test_file)
    
    for train_lang, lang in [(train_pt, 'pt'), (train_es, 'es')]:
        print('starting fasttext for ', lang)
        start_time = time.time()
        model = gensim.models.FastText(train_lang, min_count=5, size=200, workers=workers, window=8, sg=1)
        if not os.path.exists(path+'embeddings'):
            os.mkdir(path+'embeddings')
        dump(model, path+'embeddings/fasttext_sg_8_tri_'+lang+'.dump')
        #model.save(path+'fasttext_sg_'+lang+'.model')
        del model
        gc.collect()
        print("FastText time: --- %s seconds ---" % (time.time() - start_time))

def try_get_vector(word_vector, word):
    try:
        return word_vector.get_vector(word)
    except Exception as e:
        return None

def load_embedding_matrix(name="word2vec_sg_hs", tokenizer='keras_tokenized', lang='pt', model_type='txt'):
    print('Loading for '+ name + '_' + lang)
    model = {'pt': 'pt_core_news_sm', 'es': 'es_core_news_sm'}
    nlp = spacy.load(model[lang], disable=['tagger', 'parser', 'ner'])
    stemmer = SnowballStemmer(language=language_dic[lang])
    #spell_checker = SpellChecker(language=lang)
    #spell_checker_en = SpellChecker(language='en')
    tokenizer = load(DATA_PATH+'embeddings/'+tokenizer+'/tokenizer_'+lang+'.dump')
    model = None
    if(model_type == 'dump'):
        model = load(DATA_PATH+"embeddings/"+name+"_"+lang+'.dump')
        word_vector = model.wv
    elif(model_type == 'fast_text_bin'):
        model = gensim.models.FastText.load_fasttext_format(DATA_PATH+"embeddings/"+name+"_"+lang+'.bin')
        word_vector = model.wv
    else:
        word_vector = gensim.models.KeyedVectors.load_word2vec_format(DATA_PATH+"embeddings/"+name+"_"+lang+'.'+model_type)
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, word_vector.vector_size))
    count_unk_words = 0
    for word, i in tqdm.tqdm(tokenizer.word_index.items()):
        vector = try_get_vector(word_vector, word)
        if vector is None:
            vector = try_get_vector(word_vector, unidecode.unidecode(word))
        if vector is None:
            vector = try_get_vector(word_vector, word.capitalize())
        if vector is None:
            vector = try_get_vector(word_vector, nlp(word)[0].lemma_)
        if vector is None:
            vector = try_get_vector(word_vector, stemmer.stem(word))
        #if vector is None:
        #    vector = try_get_vector(word_vector, spell_checker.correction(word))
        #if vector is None:
        #    vector = try_get_vector(word_vector, spell_checker_en.correction(word))
        if vector is not None:
            embedding_matrix[i] = vector
        else:
            count_unk_words += 1
    del word_vector, tokenizer, model
    del nlp, stemmer#, spell_checker, spell_checker_en
    gc.collect()
    print(name+' '+lang+' unknow words: ', count_unk_words)
    return embedding_matrix
    
    
    
if __name__ == "__main__":
    fasttext(train_file=DATA_PATH+'train_preprocessed.csv', test_file=DATA_PATH+'test_preprocessed.csv', workers=6)