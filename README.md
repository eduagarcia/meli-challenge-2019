# MeLi Data Challenge 2019

## Public Leaderboard Top 1 - 0.91733

```bash
#Install requirements
pip install -r requirements.txt

#Donwload dataset to ./data folder
cd data
wget https://meli-data-challenge.s3.amazonaws.com/train.csv.gz
wget https://meli-data-challenge.s3.amazonaws.com/test.csv
gunzip train.csv.gz
cd ..

#Preprocess train and test data
python preprocessing.py

#Train FastText embedding on the preprocessed corpus
python embedding.py

#Vectorize the text corpus into a sequence of integers for the embedding layer of each deep learning model
python tokenizer.py

#Train all models used for the final inference (list on ./models folder)
python train.py --all

#Ensemble the results of all models and generate the submission file (./data/submission.csv)
python ensemble.py

```

MercadoLibre Data Challenge 2019
