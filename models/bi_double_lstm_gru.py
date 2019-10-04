from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras_metrics
import util
import keras
from parameters import DATA_PATH, padding_len
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.utils import to_categorical
import os
from dataset import DataGenerator
from joblib import load, dump
from keras.metrics import categorical_accuracy
from keras.preprocessing.sequence import pad_sequences
from embedding import load_embedding_matrix
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Reshape, Flatten
import numpy as np
import pandas as pd
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, TimeDistributed, MaxPooling1D, BatchNormalization
import gc
from imblearn import FunctionSampler
from keras.layers import Concatenate, Input, Dropout, SpatialDropout1D
from keras.models import Model
from attention import SeqWeightedAttention, Attention
from util import CyclicLR
from sklearn.model_selection import StratifiedShuffleSplit
import keras.backend as K

NAME = "bi_double_lstm_gru"

PARAMS = {
    'sequence_len': padding_len,
    'embedding_dim': 200,
    'epochs': 2,
    'batch_size': 256,
    'loss': 'categorical_crossentropy',
    'num_classes': len(util.get_categories()),
    'class_weights': None,
    'sampler': None,
    'k-folds': 2
}

PATH = DATA_PATH+'models/'+NAME+'/'

DEPENDENCIES = {
                    'categorical_recall': keras_metrics.categorical_recall(),
                    'balanced_recall': util.balanced_recall,
                    'SeqWeightedAttention': SeqWeightedAttention,
                    'Attention': Attention,
                    'f1': util.f1
                }

def load_model(path, extras={}):
    dependencies = {**DEPENDENCIES, **extras}
    return keras.models.load_model(path, custom_objects=dependencies)

def load_lastest(lang='pt', extras={}):
    if (len(os.listdir(PATH)) > 0):
            highest = (0, '')
            for file in os.listdir(PATH):
                if(file.startswith('weights') and file.endswith(lang+'.hdf5')):
                    epoch = int(file.split('-')[1])
                    if(highest[0] < epoch):
                        highest = (epoch, file)
            if(highest[0] > 0):
                model = load_model(PATH+highest[1], extras=extras)
                epoch = highest[0]
                return model, epoch
    return None
    
def generate_model(params):
    inputs = Input(shape=(PARAMS['sequence_len'],), dtype='int32')
    input_layer = Embedding(input_dim=params['vocab_size'],
                        output_dim=params['embedding_dim'],
                        input_length=PARAMS['sequence_len'],
                        weights=[params['embedding_matrix']],
                        trainable=False)(inputs)
    i1 = Bidirectional(CuDNNLSTM(params['embedding_dim']*2, return_sequences=True))(input_layer)
    i1 = CuDNNLSTM(params['embedding_dim'])(i1)
    i2 = Bidirectional(CuDNNGRU(params['embedding_dim'], return_sequences=True))(input_layer)
    i2 = CuDNNGRU(params['embedding_dim']//2)(i2)
    concatenated_tensor = Concatenate(axis=1)([i1, i2])
    #concatenated_tensor = Dense(params['num_classes']*2, activation = 'relu')(concatenated_tensor)
    #concatenated_tensor = BatchNormalization()(concatenated_tensor)
    #concatenated_tensor = Dropout(0.1)(concatenated_tensor)
    output = Dense(params['num_classes'], activation="softmax")(concatenated_tensor)

    model = Model(inputs=inputs, outputs=output)

    opt=Adam()

    model.compile(optimizer=opt, loss=params['loss'],
                  metrics=['accuracy',
                           categorical_accuracy,
                           keras_metrics.categorical_recall(),
                           util.balanced_recall,
                           util.f1
                           ])

    model.summary()

    return model, params

def process_y(y):
        return to_categorical(y, num_classes=PARAMS['num_classes'])
    
def process_x(x):
        return pad_sequences(x, maxlen=PARAMS['sequence_len'])
    
def balance_dataset(X, y, cut_off=0.5, random_state=42):
    xyconcat = np.concatenate((X, y.reshape(-1,1)), axis=1)
    xyconcat.view('i8,i8').sort(order=['f1'], axis=0)
    unique_y, freq = np.unique(xyconcat[:, 1], return_counts=True)
    unique_y_dict = {y: i for i, y in enumerate(unique_y)}
    unique_y_sorted = unique_y[freq.argsort()[::-1]]
    dict_sorted = np.asarray([unique_y_dict[i] for i in unique_y_sorted])
    split_sorted = np.asarray(np.split(xyconcat, np.cumsum(freq)[:-1]))[dict_sorted]
    X1, y1 = np.hsplit(np.concatenate(split_sorted[:int(len(split_sorted)*cut_off)]), 2)
    X2, y2 = np.hsplit(np.concatenate(split_sorted[int(len(split_sorted)*cut_off):]), 2)
    X1, y1 = RandomUnderSampler(random_state=random_state).fit_resample(X1, y1)
    X2, y2 = RandomOverSampler(random_state=random_state).fit_resample(X2, y2)
    X_sampled = np.concatenate((X1, X2))
    y_sampled = np.concatenate((y1.reshape(-1), y2))
    del xyconcat, split_sorted
    del X1, y1, X2, y2
    gc.collect()
    return X_sampled, y_sampled

def train(lang='pt'):
    params = PARAMS.copy()
    initial_epoch = 0
    X, Y = util.get_X_Y(data_type='keras_tokenized_tri', lang=lang, file_type="dump")
    X = np.asarray(X)
    params['embedding_matrix'] = load_embedding_matrix(name="fasttext_sg_tri_8", tokenizer='keras_tokenized_tri',lang=lang, model_type="dump")
    params["vocab_size"] = params['embedding_matrix'].shape[0]
    params["embedding_dim"] = params['embedding_matrix'].shape[1]
    
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    if not os.path.exists(PATH+'log_dir'):
        os.makedirs(PATH+'log_dir')
    
    kfold_count = 1
    skf = StratifiedShuffleSplit(n_splits=params['k-folds'], test_size=200000)
    for train_index, test_index in skf.split(X, Y):
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        X_train, Y_train = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index]
        
        params['sampler'] =  FunctionSampler(func=balance_dataset,
                          kw_args={'cut_off': 0.5,
                                  'random_state': np.random.randint(0, 100000)})
        
        model, params = generate_model(params)
        
        print(model.metrics_names)
        
        data_g_train = DataGenerator(X_train, Y_train, lang=lang, process_x=process_x, process_y=process_y, sampler=params['sampler'], batch_size=PARAMS['batch_size'], separate_val=False)
        data_g_val = DataGenerator(X_test, Y_test, lang=lang, process_x=process_x, process_y=process_y, batch_size=PARAMS['batch_size'], separate_val=False)
        print('data_generator.x: ', data_g_train.__getitem__(0)[0][0:5])
        print('data_generator.y: ', data_g_train.__getitem__(0)[1][0:5])

        #params["class_weights"]= data_generator.get_classes_weights()

        reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.2, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.02, patience=10, verbose=1)
        csv_logger = CSVLogger(PATH+'traning.log', append=True)
        tensorboard_callback = TensorBoard(log_dir=PATH+'log_dir', batch_size=params["batch_size"])
        model_checkpoint = ModelCheckpoint(filepath=PATH+'weights-{epoch:03d}-{val_categorical_accuracy:.4f}-'+lang+'.hdf5',
                                                   monitor='val_categorical_accuracy',
                                                   verbose=1,
                                                   mode='max')
        clr = CyclicLR(base_lr=1e-3, max_lr=2e-3,
                   step_size=300., mode='exp_range',
                   gamma=0.99994)

        params["callbacks"] = [tensorboard_callback, csv_logger, clr]
        
        for epoch_i in range(params["epochs"]):
            model.fit_generator(data_g_train,
                                epochs=1,
                                verbose=1,
                                callbacks=params["callbacks"],
                                #workers=7,
                                #use_multiprocessing=True,
                                class_weight=params["class_weights"])

            batch_val = 1000
            data_g_val.set_batch_size(batch_val)
            y_pred = np.zeros(Y_test.shape)
            y_val = np.zeros(Y_test.shape)
            for i, (x, y) in enumerate(data_g_val):
                y_pred[i*batch_val:(i+1)*batch_val] = np.argmax(model.predict(x), axis=1)
                y_val[i*batch_val:(i+1)*batch_val] = np.argmax(y, axis=1)
            result = util.evaluate(y_val, y_pred)
            print('Model '+NAME+' val score on '+lang+', k-fold-'+str(kfold_count)+': ', result)
            model.save(PATH+'weights-{epoch:03d}-{result:.4f}-{lang}.hdf5'.format(epoch=epoch_i+1, result=result, lang=lang))
        del data_g_train, data_g_val
        del model
        del X_train, Y_train, X_test, Y_test
        K.clear_session()
        gc.collect()
        kfold_count += 1
        break
    
def evaluate(lang='pt'):
    X, Y = util.get_X_Y(data_type='keras_tokenized_tri', lang=lang, file_type="dump")
    X = np.asarray(X)
    data_generator = DataGenerator(X,Y, lang=lang, process_x=process_x, batch_size=PARAMS['batch_size'])
    model, epoch = load_lastest(lang=lang)
    x_val, y_val = data_generator.get_validation_data()
    y_pred = model.predict(x_val)
    y_pred = y_pred.argmax(axis=-1)
    print('Model '+NAME+' val score on '+lang+': ', util.evaluate(y_val, y_pred))
    
def generate_result_file():
    dsets = []
    for lang in ['pt', 'es']:
        X = util.get_X_test(data_type='keras_tokenized_tri', lang=lang, file_type="dump")
        model, epoch = load_lastest(lang=lang)
        y_pred = util.one_hot_decode(model.predict(process_x(X)))
        index = np.load('./data/test_index_'+lang+'.npy')
        df = pd.DataFrame({'id': index, 'category': y_pred})
        df.index = df['id']
        dsets.append(df)
        print('y_pred '+lang+' unique: ', len(np.unique(y_pred)))
    df = pd.concat(dsets)
    df = df.sort_index()
    df[['id', 'category']].to_csv('./data/results-'+NAME+'.csv', index=False)