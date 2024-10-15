# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:24:08 2024

@author: SAGNIK GHOSHAL
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Input,MaxPool1D,Conv2D
import chess as c
def position_parser(position_string):
    
    piece_map = {'K':[1,0,0,0,0,0,0,0,0,0,0,0],
                 'Q':[0,1,0,0,0,0,0,0,0,0,0,0],
                 'R':[0,0,1,0,0,0,0,0,0,0,0,0],
                 'B':[0,0,0,1,0,0,0,0,0,0,0,0],
                 'N':[0,0,0,0,1,0,0,0,0,0,0,0],
                 'P':[0,0,0,0,0,1,0,0,0,0,0,0],
                 'k':[0,0,0,0,0,0,1,0,0,0,0,0],
                 'q':[0,0,0,0,0,0,0,1,0,0,0,0],
                 'r':[0,0,0,0,0,0,0,0,1,0,0,0],
                 'b':[0,0,0,0,0,0,0,0,0,1,0,0],
                 'n':[0,0,0,0,0,0,0,0,0,0,1,0],
                 'p':[0,0,0,0,0,0,0,0,0,0,0,1]}
    
    position_array = []
    
    ps = position_string.replace('/','')
    
    
    for char in ps:
        position_array += 12 * int(char) * [0] if char.isdigit() else piece_map[char]
    
    #print("position_parser =>  position_array: {}".format(asizeof.asizeof(position_array)))
    
    return position_array

def fen_to_binary_vector(fen):
    
    #counter += 1
    #clear_output(wait=True)
    #print(str(counter)+"\n")
    
    fen_infos = fen.split()
    
    pieces_ = 0
    turn_ = 1
    castling_rights_ = 2
    en_passant_ = 3
    half_moves_ = 4
    moves_ = 5
    
    binary_vector = []
    
    binary_vector += ( [1 if fen_infos[turn_] == 'w' else 0]
                        + [1 if 'K' in fen_infos[castling_rights_] else 0]
                        + [1 if 'Q' in fen_infos[castling_rights_] else 0]
                        + [1 if 'k' in fen_infos[castling_rights_] else 0]
                        + [1 if 'q' in fen_infos[castling_rights_] else 0]
                        + position_parser(fen_infos[pieces_])
                        )
    
    #print("fen_to_binary_vector =>  binary_vector: {}".format(asizeof.asizeof(binary_vector)))
    #clear_output(wait=True)
    
    return binary_vector
df=pd.read_csv("fen.csv")
df.columns=['fen','value']
print(df.shape)
print(c.Board(df['fen'][0]))
df1=pd.DataFrame()
bivec=[]
n=100000
for i in range(n):
    bivec.append(fen_to_binary_vector(df['fen'][i]))
    if i%10000==0:
        print(i)
df1['val']=df['value'][:n]
df1['bivec']=bivec
print(type(df1['bivec'][0]),df1['bivec'][0])
X=np.array(list(map(np.array,df1['bivec'])))
y=np.array(df1['val'])
scaler = MinMaxScaler()
print(X.ndim)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
scaler.fit(y_train.reshape(-1, 1))
y_train = scaler.transform(y_train.reshape(-1, 1))
#dump(scaler,"scaler.joblib")
print(X_train)
model = Sequential()
model.add(Input(shape=(773,)))
#model.add(Dense(10000,  activation='relu'))
#model.add(Dense(2000,  activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=5,batch_size=64)
plt.plot(history.history['loss'])
plt.title('Model loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.figure(figsize=(16, 8))
y_pred=scaler.inverse_transform(model.predict(X_test))
y_tpred=scaler.inverse_transform(model.predict(X_train))
# plotting training and test
plt.plot([i for i in range(len(y_test))],y_test, label="actual values", c='r')
plt.plot([i for i in range(len(y_test))],y_pred, label="Predicted values", c='g')
for i in range(100):
    print(y_test[i],y_pred[i])
# showing the plotting
plt.legend()
plt.show()
# Importing the evaluation metrics
from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_percentage_error
# R-score --> evaluation metrics
print('R2 score is :', r2_score(y_test,y_pred),'rmse score is :',root_mean_squared_error(y_test,y_pred),'mape score is :',mean_absolute_percentage_error(y_test,y_pred))
print('R2 score is :', r2_score(scaler.inverse_transform(y_train),y_tpred),'rmse score is :',root_mean_squared_error(scaler.inverse_transform(y_train),y_tpred)
      ,'mape score is :',mean_absolute_percentage_error(scaler.inverse_transform(y_train),y_tpred))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
bivec=[]
for i in range(n,2*n):
    bivec.append(fen_to_binary_vector(df['fen'][i]))
    if i%1000==0:
        print(i)
y_pred=scaler.inverse_transform(model.predict(tf.convert_to_tensor(list(map(tf.convert_to_tensor,bivec)))))
y_test=np.array(df['value'][n:2*n])
print('R2 score is :', r2_score(y_test,y_pred),'rmse score is :',root_mean_squared_error(y_test,y_pred),'mape score is :',mean_absolute_percentage_error(y_test,y_pred))
plt.plot([i for i in range(1000)],y_test[:1000], label="actual values", c='r')
plt.plot([i for i in range(1000)],y_pred[:1000], label="Predicted values", c='g')
plt.show()




