#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:57:39 2022

@author: muratkuzlu
"""
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM, GRU, Flatten, Dropout
from keras.layers import ConvLSTM2D
from keras.layers import Bidirectional
import keras_tuner as kt
from keras.layers import RepeatVector

import sklearn.metrics as metrics
import csv
import os.path

################################################
# We cannot fit the model like we normally do for image processing where we have
#X and Y. We need to transform our data into something that looks like X and Y values.
# This way it can be trained on a sequence rather than indvidual datapoints. 
# Let us convert into n number of columns for X where we feed sequence of numbers
#then the final column as Y where we provide the next number in the sequence as output.
# So let us convert an array of values into a dataset matrix

#seq_size is the number of previous time steps to use as 
#input variables to predict the next time period.

#creates a dataset where X is the number of passengers at a given time (t, t-1, t-2...) 
#and Y is the number of passengers at the next time (t + 1).

PERFORMANCE_TEST_DICT = {
    "Model" : 0,
    "RMSE"      : 0,
    "MSE"        : 0,
    "MAE"		  : 0,
    "CTime"		  : 0,
    "Eps" 	  : 0,
}

#Attention
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)

def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)

################################################
#Adversarial
def my_fgsm(x_input, y_output, DL_model, eps):
    x_input_tensor = tf.cast(x_input, tf.float32)
    loss_object = tf.keras.losses.MeanSquaredError()

    with tf.GradientTape() as tape:
        tape.watch(x_input_tensor)
        prediction = DL_model(x_input_tensor)

        loss = loss_object(y_output, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, x_input_tensor)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    #return x_input + signed_grad*eps/len(x_input)
    return x_input + signed_grad*eps/len(x_input[0][0])
   #return x_input + np.random.uniform(0,eps/len(x_input[0][0]),len(x_input[0][0]))
################################################
################################################
#Calculate performcance metrics
def calc_perf_metrics(test, prediction):
    mse= metrics.mean_squared_error(test, prediction)
    rmse = np.sqrt(mse) 
    mae=metrics.mean_absolute_error(test, prediction)
    r2 = metrics.r2_score(test, prediction)

        
    return rmse, mse, mae, r2
#################################################
#################################################
#Create RRN Model
def create_RNN_model(trainX, trainY, seq_size):
    model = Sequential()
    model.add(SimpleRNN(100, activation='relu', return_sequences=True,  input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(SimpleRNN(100, activation='relu'))
    model.add(Dense(32))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
        
    return model
#################################################
#################################################
#Create LSTM Model
def create_LSTM_model(trainX, trainY, seq_size):
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(32))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
        
    return model
#################################################
#################################################
#Create BiLSTM Model
def create_BiLSTM_model(trainX, trainY, seq_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(trainX.shape[1], trainX.shape[2])))
    #model.add(Bidirectional(LSTM(100, activation='relu')))
    #model.add(Dense(32))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
        
    return model
#################################################
#################################################
#Create GRU Model
def create_GRU_model(trainX, trainY, seq_size):
    model= Sequential()
    model.add(GRU(100, activation='tanh', return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(GRU(100, activation='tanh'))
    model.add(Dense(32))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
        
    return model
#################################################

#################################################
#Create LSTM_Attention Model
def create_LSTM_Attention_model(trainX, trainY, seq_size):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    #model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Attention(return_sequences=False)) # receive 3D and output 3D
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mse')
        
    return model
#################################################
#################################################
#Create BiLSTM_Attention Model
def create_BiLSTM_Attention_model(trainX, trainY, seq_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True), input_shape=(trainX.shape[1], trainX.shape[2])))
    #model.add(Bidirectional(LSTM(64, activation='relu')))
    model.add(Attention(return_sequences=False)) # receive 3D and output 3D
    #model.add(Bidirectional(LSTM(100, activation='relu')))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
   
        
    return model
#################################################
#################################################
#Create BiLSTM_Attention Model
def create_LSTM_Encoder_model(trainX, trainY, seq_size):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    #model.add(Dropout(rate=0.2))
    #model.add(RepeatVector(trainX.shape[1]))
    model.add(RepeatVector(trainY.shape[1]))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(100, activation='relu', return_sequences=False))
    #model.add(Dropout(rate=0.2))
    #model.add(TimeDistributed(Dense(trainY.shape[2])))
    model.add(Dense(trainY.shape[1]))

    
    model.compile(optimizer='adam', loss='mse')
        
    return model
#################################################
#################################################
#################################################
#Write file
def write_file(model, rmse, mse, mae, esp, comptime, file_name):


    PERFORMANCE_TEST_DICT['Model'] = model
    PERFORMANCE_TEST_DICT['RMSE'] = rmse
    PERFORMANCE_TEST_DICT['MSE'] = mse
    PERFORMANCE_TEST_DICT['MAE'] = mae
    PERFORMANCE_TEST_DICT['CTime'] = comptime
    PERFORMANCE_TEST_DICT['Eps'] = esp
    
    with open(file_name, mode='a') as csv_file:
     
     writer = csv.DictWriter(csv_file, PERFORMANCE_TEST_DICT.keys())
     writer.writerow(PERFORMANCE_TEST_DICT)
        
    return model
#################################################


