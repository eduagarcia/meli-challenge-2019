import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from parameters import DATA_PATH, language_dic, validation, file_posfix
from tokenizer import SimpleVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from joblib import load, dump
import os
import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
import pickle
import keras

def get_vec(data_type="counter", lang='pt'):
    return load(DATA_PATH+data_type+"_vec_"+lang)

def get_X_Y(data_type='tfidf', lang='pt', file=None, y_file=None, file_type="npy"):
    if(file == None):
        file = 'train'+file_posfix+'.csv'
    usecols=["language"]
    if(data_type == "text"):
        usecols.append("title")
    if(y_file == None):
        usecols.append("category")
    if(y_file):
        y = pd.read_csv(DATA_PATH+y_file, usecols=["category"])
        df["category"] = y["category"]
    if(data_type == 'tfidf' or data_type == 'counter'):
        train = load(DATA_PATH+data_type+'_trans_train_'+lang)
    elif(data_type == "text"):
        df = pd.read_csv(DATA_PATH+file, usecols=usecols)
        df = df[df["language"] == language_dic[lang]]
        def to_str(X):
            for i, j in enumerate(X):
                X[i] = str(j)
            return X
        train = to_str(df["title"].values)
    else:
        if(file_type == 'npy'):
            train = np.load(DATA_PATH+'embeddings/'+data_type+'/'+lang+'.npy')
        else:
            train = load(DATA_PATH+'embeddings/'+data_type+'/'+lang+'.dump')
    return train, np.load(DATA_PATH+'categories_int_'+lang+'.npy')

def get_X_test(data_type='text', lang='pt', file=None, file_type="npy"):
    if(file == None):
        file = 'test'+file_posfix+'.csv'
    usecols=["language"]
    if(data_type == "text"):
        usecols.append("title")
        df = pd.read_csv(DATA_PATH+file, usecols=usecols)
        df = df[df["language"] == language_dic[lang]]
    if(data_type == 'tfidf' or data_type == 'counter'):
        train = load(DATA_PATH+data_type+'_trans_train_'+lang)
    elif(data_type == "text"):
        def to_str(X):
            for i, j in enumerate(X):
                X[i] = str(j)
            return X
        train = to_str(df["title"].values)
    else:
        if(file_type == 'npy'):
            train = np.load(DATA_PATH+'embeddings/'+data_type+'/'+lang+'_test.npy')
        else:
            train = load(DATA_PATH+'embeddings/'+data_type+'/'+lang+'_test.dump')
    return train

def get_categories():
    with open(DATA_PATH+'categories.txt', 'r') as file:
        return list(map(lambda x: x.replace('\n', ''), file.readlines()))

def int_encode(Y):
    label_encoder = LabelEncoder().fit(get_categories())
    return label_encoder.transform(Y)

def one_hot_encode(Y):
    return to_categorical(int_encode(Y), num_classes=len(get_categories()))

def one_hot_decode(Y):
    return int_decode(Y.argmax(axis=-1))

def int_decode(Y):
    label_encoder = LabelEncoder().fit(get_categories())
    return label_encoder.inverse_transform(Y)

NUM_CLASSES = len(get_categories())
def to_categorical(Y):
    return keras.utils.to_categorical(Y, num_classes=NUM_CLASSES)

def evaluate(Y_true, Y_pred):
    return balanced_accuracy_score(Y_true, Y_pred)

import keras.backend as K

def balanced_recall(y_true, y_pred):
    """
    Computes the average per-column recall metric
    for a multi-class classification problem
    """ 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)  
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)   
    recall = true_positives / (possible_positives + K.epsilon())    
    balanced_recall = K.sum(recall, axis=0)/K.cast(K.shape(recall)[0], recall.dtype)
    return balanced_recall

def focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return focal_loss_fixed

class ModelCheckpointEnhanced(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        # Added arguments
        self.callbacks_to_save = kwargs.pop('callbacks_to_save')
        self.callbacks_filepath = kwargs.pop('callbacks_filepath')
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        # Run normal flow:
        super().on_epoch_end(epoch,logs)

        # If a checkpoint was saved, save also the callback
        filepath = self.callbacks_filepath.format(epoch=epoch + 1, **logs)
        if self.epochs_since_last_save == 0 and epoch!=0:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current == self.best:
                    # Note, there might be some cases where the last statement will save on unwanted epochs.
                    # However, in the usual case where your monitoring value space is continuous this is not likely
                    with open(filepath, "wb") as f:
                        dump(self.callbacks_to_save, f)
            else:
                with open(filepath, "wb") as f:
                    dump(self.callbacks_to_save, f)

# https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate/code

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
    

def f1(y_true, y_pred):
    '''
    metric from here 
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
