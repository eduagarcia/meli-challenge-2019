import numpy as np
import util
import attention
import keras_metrics
from dataset import DataGenerator
from keras.preprocessing.sequence import pad_sequences
import importlib
importlib.reload(util)
import gc
from parameters import DATA_PATH, padding_len
import os
import keras
import keras.backend as K
import tqdm
import pandas as pd

def process_y(y):
    return util.to_categorical(y)

def process_x(x):
    return pad_sequences(x, maxlen=padding_len)

if __name__ == "__main__":
    
    VERSION = 'ENSEMBLE'
    #confidence of each network
    model_weights_int = {
            'bi_double_lstm_gru': 3,
            'bi_lstm_cnn': 2,
            'bi_lstm_gru_attw_dense': 3,
            'bi_lstm_gru_balanced': 3,
            'bi_lstm_gru_selfatt_kfold': 5,
            'bi_lstm_gru_spat_clr': 6,
            'bi_lstm_gru_spat_clr_kfold': 5,
            'text_cnn_att': 1
            }
    #weights normalized
    model_weights = {k:(v/45) for k, v in model_weights_int.items()}
    model_list = [k for k in model_weights_int.keys()]
    
    #weights for each epoch according with the number of epochs trained
    weigths_epoch = {
                1: [1],
                2: [0.35, 0.65],
                3: [0.15, 0.35, 0.5],
                4: [0.1, 0.2, 0.3, 0.4], 
                5: [0.1, 0.15, 0.2, 0.25, 0.3]} 

    num_classes = len(util.get_categories())
    
    #Load test data for each language
    data = {}
    for lang in ['es', 'pt']:
        X_test = util.get_X_test(data_type='keras_tokenized_tri', lang=lang, file_type="dump")
        index = np.load(DATA_PATH+'test_index_'+lang+'.npy')
        data[lang] = {'index': index, 'X_test': process_x(X_test)}
        del X_test, index
        gc.collect()

    paths = {}
    for model_name in model_list:
        PATH = DATA_PATH+'models/'+model_name+'/'
        files = {'pt': {}, 'es': {}}
        if (len(os.listdir(PATH)) > 0):
            for file in os.listdir(PATH):
                if(file.startswith('weights')):
                    epoch = int(file.split('-')[1])
                    lang = file.split('-')[-1].split('.')[0]
                    if('kfold' in model_name):
                         epoch = int(file.split('-')[2][-1])
                    files[lang][epoch]= PATH+file
        paths[model_name] = files

    #results_int = {}
    general_result = np.zeros((data['es']['X_test'].shape[0]+data['pt']['X_test'].shape[0] ,num_classes))
    K.clear_session()
    for model_name in tqdm.tqdm(model_list):
        print(model_name)
        result_model = np.zeros((data['es']['X_test'].shape[0]+data['pt']['X_test'].shape[0] ,num_classes))
        for lang in ['es', 'pt']:
            n_epochs = len(paths[model_name][lang])
            result_model_lang = np.zeros((data[lang]['X_test'].shape[0] ,num_classes))
            for epoch in range(n_epochs):
                epoch = epoch+1
                #print(paths[model_name][lang][epoch])
                DEPENDENCIES = {
                    'categorical_recall': keras_metrics.categorical_recall(),
                    'balanced_recall': util.balanced_recall,
                    'AttentionWeightedAverage': attention.AttentionWeightedAverage,
                    'Attention': attention.Attention,
                    'SeqSelfAttention': attention.SeqSelfAttention,
                    'SeqWeightedAttention': attention.SeqWeightedAttention,
                    'f1': util.f1
                }
                model = keras.models.load_model(paths[model_name][lang][epoch], custom_objects=DEPENDENCIES)
                result = model.predict(data[lang]['X_test'])
                weight = 1
                if('kfold' in model_name):
                    weight = 1/n_epochs
                else:
                    weight = weigths_epoch[n_epochs][epoch-1]
                result_model_lang = result_model_lang + (result * weight)
                del model, result
                gc.collect()
                K.clear_session()
            result_model[data[lang]['index']] = result_model_lang
            del result_model_lang
            gc.collect()
        general_result = general_result + result_model * model_weights[model_name]                                                                             
        #results_int[model_name] = np.argmax(result_model, axis=1)
        del result_model
        gc.collect()
    
    target_file = DATA_PATH+'submission.csv'
    print('Writing result to ', target_file)
    final_result_df = pd.DataFrame({'id': np.arange(general_result.shape[0]), 'category': util.int_decode(np.argmax(general_result, axis=1))})
    final_result_df[['id', 'category']].to_csv(target_file, index=False)