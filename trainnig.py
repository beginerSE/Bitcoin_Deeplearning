# ライブラリのインポート
# import tensorflow.python.keras
from keras.layers.recurrent import SimpleRNN
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import time
import datetime
import requests
import json
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Flatten



def get_bitcoinprice():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=jpy&days=max'
    r = requests.get(url)
    r = json.loads(r.text)
    bitcoin = r['prices']
    data, date = [], []
    for i in bitcoin:
        data.append(i[1])
        date.append(i[0])
    bitcoin = pd.DataFrame({"date":date, "price":data})
    price = bitcoin['price']
    change = price.pct_change()
    bitcoin = pd.DataFrame({"date":date,"price":data,"change":change})
    return bitcoin


# 価格データをディープラーニング用に整形する
length = len(bitcoin['change'])
term = 30
pricedata = []
answer = []
for i in range(0, length-term):
    pricedata.append(bitcoin['change'].iloc[i:i+term])
    answer.append(bitcoin['change'].iloc[i+term])


# データを9:1に分割する
x = np.array(pricedata).reshape(len(pricedata), term, 1)
y = np.array(answer).reshape(len(answer), 1)

y2 = []
for i in y:
    if i > 0:
        y2.append(1)
    else:
        y2.append(0)

(X_train, X_test,y_train, y_test) = train_test_split(x, np.array(y2), test_size=0.1, random_state=0, shuffle=False)


# ベクトルから1 hot に変換(正解を列にして表示する作業)
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)


#========Kerasでのディープラーニング==========#

model = Sequential()
model.add(LSTM(512, kernel_initializer='random_normal',input_shape=(30, 1)))


# 20%(全体の重みの何％を消去するか)をドロップするDropOut層を追加
model.add(Dropout(0.2))

# 出力を512次元とする、重みの初期値をザビエルとした全結合層 (Dense)とReLU層を追加
model.add(Dense(512, activation='relu',kernel_initializer='glorot_normal'))

# 20%をドロップするDropOut層を追加
model.add(Dropout(0.2))

# 出力を512次元とする、重みの初期値をザビエルとした全結合層 (Dense)とReLU層を追加
model.add(Dense(512, activation='relu',kernel_initializer='glorot_normal'))

# 20%をドロップするDropOut層を追加
model.add(Dropout(0.2))

# 出力を512次元とする、重みの初期値をザビエルとした全結合層 (Dense)とReLU層を追加
model.add(Dense(512, activation='relu',kernel_initializer='glorot_normal'))

# 20%をドロップするDropOut層を追加
model.add(Dropout(0.2))

# 出力を512次元とする、重みの初期値をザビエルとした全結合層 (Dense)とReLU層を追加
model.add(Dense(512, activation='relu',kernel_initializer='glorot_normal'))

# 20%をドロップするDropOut層を追加
model.add(Dropout(0.2))

# 出力を512次元とする、重みの初期値をザビエルとした全結合層 (Dense)とReLU層を追加
model.add(Dense(512, activation='relu',kernel_initializer='glorot_normal'))

model.add(Dense(2, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
history=model.fit(X_train,y_train, epochs=10, batch_size=128, verbose=1,validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print(score[1])


# ==========学習過程の可視化==========

#Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
