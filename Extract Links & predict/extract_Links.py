from  urllib.request import urlopen
import requests
import csv
from datetime import datetime
from bs4 import BeautifulSoup
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model 
data = []

# specify the url
quote_page ="http://localhost/p/public/master.php?page=login.php"

page =requests.get(quote_page).text
soup = BeautifulSoup(page, "html.parser")
for link in soup.find_all('a'):
    data.append(link.get('href')) 
# extract link from master login page
website_domain_name = "localhost"
#open a csv file with append, so old data will not be erased
with open("phishing_test.csv", "a" , newline ="" ) as csv_file:
        writer = csv.writer(csv_file)
        for url in data:
            if website_domain_name not in url:
                writer.writerow([url])

#Testing urls 
features=[]
with open("phishing_test.csv" ,'r') as f:
  data = csv.reader(f)
  for url in data:
      features.append(url)
print(features)
#know model word index and tokenizer from original data
X=[]
Y=[]
with open("last-data.csv" ,'r') as f:
  data = csv.reader(f)
  for row in data:
    url= row[0]
    label= row[1]
    X.append(url)
    Y.append(label)
    
train_size =int(len(X) * 0.75)
X_train = X[:train_size]

# Tokenize training data extract dictionary only from training data
tokenizer = Tokenizer(filters='\t\n', char_level=True)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
num_words = len(word_index) + 1


# processing test data
max_length = 2083
test_url=features
test = tokenizer.texts_to_sequences(test_url)
test = pad_sequences(test, maxlen=max_length , padding = 'pre')
test = test.astype(float)
#Load the trained LSTM model
weights_file = 'lstm-weights.h5'
model =load_model ('lstm-model.h5')
model.load_weights(weights_file)
predicted = model.predict(test, verbose=1)
for p in predicted:
   if p> 0.5:
      p="Phishing"
   else:
      p="Benign"
   print(p)
#open a csv file with append, and update testing data with label
with open("phishing_test_label.csv", "a" , newline ="" ) as csv_file:
        writer = csv.writer(csv_file)
        for url in test_url:
            writer.writerow([url[0] , p])

