import h5py
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
import math
import numpy as np
from parameters import DATA_PATH, file_posfix, language_dic, padding_len, validation
import gc
import util
from joblib import load

class DataGenerator(Sequence):
    def __init__(self, x, y, lang='pt', batch_size=32, process_x=lambda x: x, process_y=lambda y: y,separate_val=validation, sampler=None):
        #self.h5 = h5py.File(h5_file, 'r', libver='latest')['test']
        self.lang = lang
        #self.lang_dset = self.h5[lang]
        self.process_x = process_x
        self.process_y = process_y
        self.num_classes = len(util.get_categories())
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.separate_val = separate_val
        if(self.separate_val):
            self.val_indices = np.load(DATA_PATH+'val_index_'+self.lang+'.npy')
            self.indices = np.setdiff1d(self.indices, self.val_indices)
        self.sampler = sampler
        if(self.sampler != None):
            self.indices, y =  self.sampler.fit_resample(self.indices.reshape(-1,1), self.y[self.indices])
            del y
            gc.collect()
            self.indices = self.indices.reshape(-1)
        np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(self.indices.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        #inds.sort()
        batch_x = self.process_x(self.x[inds])
        batch_y = self.process_y(self.y[inds])
        return batch_x, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        
    def get_validation_data(self):
        if(self.separate_val):
            x = self.process_x(self.x[self.val_indices])
            y = self.process_y(self.y[self.val_indices])
            return x, y
        else:
            return None
        
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size