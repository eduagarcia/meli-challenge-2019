from models import MODEL_DICT
from parameters import DATA_PATH
import os

def train_model(model):
    if not os.path.exists(DATA_PATH+'models'):
        os.mkdir(DATA_PATH+'models')
    print('Starting traning of ', model.NAME)
    model.train(lang='pt')
    model.train(lang='es')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, required=False, default='bi_lstm_gru_spat_clr',
                        help='model name')
    parser.add_argument('--all', default=False, action='store_true',
                        help='train all models')

    args = parser.parse_args()
    if(args.all):
        for model in MODEL_DICT.values():
            train_model(model)
    else:
        train_model(MODEL_DICT[args.model])