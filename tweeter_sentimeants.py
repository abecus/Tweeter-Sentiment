#%%
import pickle
from nltk.stem.porter import PorterStemmer

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

#%%
sentiments_path = 'Sentiment.csv'
df = pd.read_csv(sentiments_path, delimiter=',')
print(df.head(5))

#%%
data = df[['sentiment', 'sentiment_confidence', 'text']]
df.sentiment[df.sentiment == 'Negative'] = 2
df.sentiment[df.sentiment == 'Positive'] = 1
df.sentiment[df.sentiment == 'Neutral'] = 0
print(data['sentiment'].head(10))

#%%
''' taking out the texts for preprosessing and lowering
    the strings making a list of only textwords for text given '''
df['text'] = df['text'].str.lower()

porter_stemmer = PorterStemmer()
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = (porter_stemmer.stem(token) for token in tokens)
    return ' '.join(stemmed_tokens)
df['feats'] = df['text'].apply(stem_sentences)

#%%df
''' picklling '''
file_Name = "stemmed_words_tweeter"
# fileObject = open(file_Name, 'wb') 
# pickle.dump(df['feats'], fileObject)   
# fileObject.close()
fileObject = open(file_Name,'rb')  
df['feats'] = pickle.load(fileObject)  

#%%
max_word = 20000
tokenizer = Tokenizer(num_words=max_word,\
    filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True, split=' ',\
    char_level=False, oov_token=None, document_count=0)

#%%
tokenizer.fit_on_texts(df['feats'].values)

#%%
X = tokenizer.texts_to_sequences(df['feats'].values)
# print((X[0]))
X = pad_sequences(X)

#%%
x_w_conf = np.column_stack((X, df['sentiment_confidence']))

#%%
input_dim = 32
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_word, embed_dim,input_length = x_w_conf.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


#%%
x_train, x_test, y_train, y_test = train_test_split(x_w_conf ,df['sentiment'], test_size = 0.33, random_state = 42)
batch_size = 32
model.fit(x_train, y_train, epochs = 10, batch_size=batch_size, verbose = 2)

#%%
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2, batch_size = batch_size)
print('Test accuracy:', test_acc,'and loss: ',  test_loss)

#%%
result = model.predict(x_test)

#%%
for i in result:
    print(i.argmax(axis=0))

#%%
model.save('twitter_sentiment_model.h5') 