# https://youtu.be/97bZKO6cJfg
"""
Dataset from: https://www.kaggle.com/rakannimer/air-passengers
International Airline Passengers prediction problem.
This is a problem where, given a year and a month, the task is to predict 
the number of international airline passengers in units of 1,000. 
The data ranges from January 1949 to December 1960, or 12 years, with 144 observations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM, GRU, Flatten, Dropout
from keras.layers import ConvLSTM2D
from keras.layers import Bidirectional
import keras_tuner as kt
from keras.layers import RepeatVector

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#from keras.callbacks import EarlyStopping

import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import os

import imp
import csv
import os.path
import time


import util
import imp

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

#imp.reload(mk_ts)



def draw_figure(eps_list, rmse_list, mse_list, mae_list, file_name):
    #Draw the figure
    fig, ax = plt.subplots(figsize=(8,4))
    
    res_df = pd.DataFrame({'Epsilon':eps_list,'RMSE':rmse_list,'MSE':mse_list,'MAE':mae_list})
    sns.lineplot(data=res_df, x='Epsilon', y='RMSE', ax=ax, color='r', marker='d')
    sns.lineplot(data=res_df, x='Epsilon', y='MSE', ax=ax, color='k', marker='v')
    sns.lineplot(data=res_df, x='Epsilon', y='MAE', ax=ax, color='b', marker='o')
    ax.set_ylabel('Values',fontsize=16,)
    ax.set_xlabel(file_name + r'Attack Power: $\epsilon$',fontsize=16,)
    ax.legend(['RMSE','MSE','MAE'], fontsize=14)
    plt.xlim([0, 1])
    plt.ylim([0, 1.0])
    plt.axhline(y=0.15, color='r', linestyle='--')
    rect1 = Rectangle((-0.1,0), 1.9, 0.15, alpha=0.2,color='green')
    rect2 = Rectangle((-0.1,0.1), 1.9, 1.9, alpha=0.1,color='red')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    plt.text(0.02, 0.03, 'Unsuccesful attacks', alpha=0.9, fontsize=20)
    plt.text(0.02, 0.4, 'Successful attacks', alpha=0.9, fontsize=20)
    
    plt.grid()
    
    plt.savefig(file_name,bbox_inches='tight')
    plt.show()
    
def draw_before_attack(y_test, y1, y2, y3, y4, y5, y6, file_name):
    in_range = 168
    x_coord = range(0,in_range)
    #x_coord_b = range(7,in_range-3)
    
    y0 = y_test[:in_range]
    y1 = y1[:in_range]
    y1[y1 < 0] = 0
    y2 = y2[:in_range]
    y2[y2 < 0] = 0
    y3 = y3[:in_range]
    y3[y3 < 0] = 0
    y4 = y4[:in_range]
    y4[y4 < 0] = 0
    y5 = y5[:in_range]
    y5[y5 < 0] = 0
    y6 = y6[:in_range]
    y6[y6 < 0] = 0
    
    plt.figure(figsize=(24,8))
    plt.ylim(0, 1.5)
    
    plt.plot( x_coord, y0, marker='', markerfacecolor='red', markersize=4, color='red', linewidth=2, label="Actual")
    plt.plot( x_coord, y1, marker='', markerfacecolor='green', markersize=4, color='green', linewidth=2, label="RNN Forecast")
    plt.plot( x_coord, y2, marker='', markerfacecolor='blue', markersize=4, color='blue', linewidth=2, label="LSTM Forecast")
    plt.plot( x_coord, y3, marker='', markerfacecolor='black', markersize=4, color='black', linewidth=2, label="BiLSTM Forecast")
    plt.plot( x_coord, y4, marker='', markerfacecolor='brown', markersize=4, color='brown', linewidth=2, label="GRU Forecast")
    plt.plot( x_coord, y5, marker='', markerfacecolor='blue', markersize=4, color='blue', linewidth=2, label="LSTM with Attention Forecast")
    plt.plot( x_coord, y6, marker='', markerfacecolor='grey', markersize=4, color='grey', linewidth=2, label="BiLSTM with Attention Forecast")
    #print(y1.shape)
    #print(y1.shape)
    #print(y1.shape)
    #plt.fill_between(x_coord_b, y0.flatten(), y1.flatten(), alpha=0.2)
    
    #plt.title("Actual vs Noise Injected Data Predictions of ML Methods", fontsize=28)
    plt.xlabel("Time Index (h)", fontsize=28)
    plt.ylabel("Solar PV Power Generation\n (Normalized)" , fontsize=28)
    
    #font = font_manager.FontProperties(family='Times New Roman',style='normal', size=20)
    plt.legend(loc='upper right', borderaxespad=0., 
            prop={"size":24 ,})
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    
    plt.grid()
    
    plt.savefig(file_name,bbox_inches='tight')
    
    plt.show()
    
    
    
###############################################
#Adversarial
###############################################
###############################################
def predict_after_attack(model_X, model_name, testX, testY, cvs_file_name, file_name_af):
    model = model_X
    ATTACK_POWER = 0.0
    rmse_list = []
    mse_list = []
    mae_list = []
    eps_list = []
    
    for ATTACK_POWER in tqdm(np.linspace(0.0,1,10).round(2)):    
        x_test_adv_fgsm = [mk_ts.my_fgsm([x_input,],[y_output,],model,ATTACK_POWER)[0] for x_input, y_output in zip(testX, testY)]
        x_test_adv_fgsm = np.array(x_test_adv_fgsm)
    
        nn_prediction_adv = model.predict(x_test_adv_fgsm)
        nn_prediction_adv[nn_prediction_adv < 0] = 0
    
        ## calculate rmse, mse, mae, r2 error
        rmse, mse, mae, r2 = mk_ts.calc_perf_metrics(testY, nn_prediction_adv)
        
        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)
        eps_list.append(ATTACK_POWER)
        comptime = 0
        eps = ATTACK_POWER
        mk_ts.write_file(model_name, rmse, mse, mae, eps, comptime, cvs_file_name)
    
    #Draw the figure
    draw_figure(eps_list, rmse_list, mse_list, mae_list, file_name_af)
###############################################

###############################################
#Adversarial Training
###############################################
def run_adversarial_traning(model_X,testX, testY):
    ADV_TRAIN_EPS_VALS = [0.01, 0.03, 0.05, 0.08, 0.1]
        
    for adv_train_val in tqdm(ADV_TRAIN_EPS_VALS, position=0, leave=True):
        adv_inputs = util.attack_models(model = model_X,attack_name = 'FGSM',
                                                eps_val = adv_train_val, testset = testX,
                                                outout_labels=testY,
                                                norm = np.inf)
        
        tmp_trainX = np.concatenate((testX,adv_inputs),axis=0)
        tmp_trainY = np.concatenate((testY,testY),axis=0)
        #print('tmp_trainX.shape',tmp_trainX.shape)
        #print('tmp_trainY.shape',tmp_trainY.shape)
        #print('    trainX.shape',trainX.shape)
        #print('    trainY.shape',trainY.shape)
        
        # load model
        model_X.fit(tmp_trainX, tmp_trainY, validation_data=(testX, testY),
            verbose=1, epochs=10)
###############################################


def run_all(params):
    
    print("Run ALL")

    result = {}
    #########################################################################
    ### CONFIGURATION

    array_attack_models = params['param_attack_models']
    attack_power = params['param_attack_power']
    array_defend_attack = params['param_defend_attack']

    #Parameters
    run_RNN = 1 #0001
    run_LSTM = 2 #0010
    run_BiLSTM = 4 #0100
    run_GRU = 8    #1000

    run_LSTM_Attention = 16    #0001 0000
    run_BiLSTM_Attention = 32    #0010 0000
    run_LSTM_Encoder = 64    #0100 0000

    grid_search_selection_string = params['grid_search_selection']
    run_GridSearch = False
    if grid_search_selection_string == "True":
        run_GridSearch = True

    run_FGSM = False    #0010 0000
    run_AdvTrain = False   #0100 0000

    #run_Models = run_RNN | run_LSTM | run_BiLSTM | run_GRU
    #run_Models = run_BiLSTM_Attention
    #run_Models = run_RNN
    run_Models = run_RNN

    run_epochs = params['param_nr_epoch'] #100
    run_batch_size = params['param_batch_size'] #32
    start_idx = 3000 #params['param_start_idx'] #3000

    df_data_from_csv = params['param_data']

    ### CONFIGURATION END
    #########################################################################
    imp.reload(util)
    
    
    plt.clf() 
    plt.close('all')
    #Change the directory
    # new_path = '/Users/muratkuzlu/Dropbox/MK_Papers/2022/2022_TimeSeries_Solar_Forecasting/Source_Code'
    # os.chdir(new_path)
    # print(os.getcwd())
    import timeseries_functions as mk_ts




    #as inputs for prediction. We only have all 5 variables until the last day in our dataset.
    n_past = 8
    seq_size = 8 
    n_future = 1
    test_size=24*7  #let us predict past 24 hours

    print("Run ALL End")

    ######################################################   
    #Initial variables
    bfattack_rmse_list = []
    bfattack_mse_list = []
    bfattack_mae_list = []
    bfattack_comp_time= []

    ######################################################   
    PERFORMANCE_TEST_DICT = {
        "Model" : 0,
        "RMSE"      : 0,
        "MSE"        : 0,
        "MAE"		  : 0,
        "CTime"		  : 0,
        "Eps" 	  : 0,
    }

    cvs_file_name = 'Model_performance_test_results.csv'
    file_exists = os.path.exists(cvs_file_name)

    if file_exists == False:
        print('File will be created')
        with open('Model_performance_test_results.csv', mode='w') as csv_file:
            #fieldnames = ['state_tur', 'state_en', 'id_no', 'lastname', 'name', 'date_of_birth', 'gender', 'doc_no', 'nationality', 'date_expiration', 'error', 'status', 'success_rate']
            # writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer = csv.DictWriter(csv_file, PERFORMANCE_TEST_DICT.keys())
            writer.writeheader()
    else:
        print('File exists')
    ######################################################   

    plt.clf() 
    plt.close('all')
    # #Change the directory
    # new_path = '/Users/muratkuzlu/Dropbox/MK_Papers/2022/2022_TimeSeries_Solar_Forecasting/Source_Code'
    # os.chdir(new_path)
    # print(os.getcwd())

    ########################################################
    # # load the dataset
    # #dataframe = read_csv('data/PV_v4.csv', usecols=[1])
    # dataframe = read_csv('data/gefcom_PV_v3.csv', usecols=[1])
    # #plt.plot(dataframe)

    # #Convert pandas dataframe to numpy array
    # dataset = dataframe.values
    # dataset = dataset.astype('float32') #COnvert values to float

    # #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # # normalize the dataset
    # scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer
    # dataset = scaler.fit_transform(dataset)

    # #We cannot use random way of splitting dataset into train and test as
    # #the sequence of events is important for time series.
    # #So let us take first 60% values for train and the remaining 1/3 for testing
    # # split into train and test sets
    # train_size = int(len(dataset) * 0.98)
    # test_size = len(dataset) - train_size
    # train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # ######################################################
    # seq_size = 6  # Number of time steps to look back 
    # #Larger sequences (look further back) may improve forecasting.

    # trainX, trainY = mk_ts.to_sequences(train, seq_size)
    # testX, testY = mk_ts.to_sequences(test, seq_size)

    # print("Shape of training set: {}".format(trainX.shape))
    # print("Shape of test set: {}".format(testX.shape))

    # # Reshape input to be [samples, time steps, features]
    # #Stacked RNN with 1 hidden dense layer
    # # reshape input to be [samples, time steps, features]
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


    # df = pd.read_csv('data/gefcom_PV_16022022.csv')
    df = df_data_from_csv
    df = df.drop(['Index'], axis=1)
    df = df.rename(columns={"Date": "Date", "Time_Index": "TI", "tcwl": "TCWL", "tciw": "TCIW", "SP": "SP", 
                            "hum": "HUM", "tcc": "TCC", "u": "U",  "v": "V", "temp": "TEMP", 
                            "TP": "TP", "SSRD": "SSRD", "STRD": "STRD", "TSR": "TSR", "Power": "POW"}, errors="raise")
    df = df.drop(['TCWL', 'TCIW', 'SP', 'U', 'V', 'SP', 'TP', 'STRD'], axis=1)
    #Separate dates for future plotting
    train_dates = pd.to_datetime(df['Date'])
    #print(train_dates.tail(15)) #Check last few dates.
    df = df[['Date', 'POW', 'TI', 'HUM', 'TEMP', 'TSR', 'SSRD']]
    df = df.dropna()

    #Separate dates for future plotting
    train_dates = pd.to_datetime(df['Date'])
    #print(train_dates.tail(15)) #Check last few dates. 
    #Variables for training
    cols = list(df)[1:7]
    #Date and volume columns are not used in training. 
    print(cols) #[''TI', 'TCC', 'TEMP', 'TSR', 'POW']
    #New dataframe with only training data - 5 columns
    df_for_training = df[cols].astype(float)
    #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = StandardScaler()
    #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer
    df_for_training_scaled = scaler.fit_transform(df_for_training)

    trainX = []
    trainY = []

    #Reformat input data into a shape: (n_samples x timesteps x n_features)
    #In my example, my df_for_training_scaled has a shape (12823, 5)
    #12823 refers to the number of data points and 5 refers to the columns (multi-variables).
    for i in range(n_past, len(df_for_training_scaled) - n_future +1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)

    #########################################################

    ###############################################
    #Attention
    ###############################################

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
        
    ###########################################################

    #########################################################
    #RNN
    ######################################################
    #RNN and Grid Search
    if ((run_Models&run_RNN) >= 1 ):
        if (run_GridSearch == True):
            print('===============RNN with Grid Search Model will run================================================')
            comptime = 0
            ##########################
            
            def model_builder(hp):
                
                hp_activation = ['relu','sigmoid','softmax','softplus','softsign','tanh','selu', 'elu']
                hp_act_func = hp.Choice('act_func', values=hp_activation)
                hp_last_act_func = hp.Choice('last_act_func', values=hp_activation)
                
                optimizer_list = ['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam']
                hp_optimizer = hp.Choice('optimizer', values=optimizer_list)
                
                hp_units1 = hp.Int('units1', min_value=10, max_value=500, step=10)
                hp_units2 = hp.Int('units2', min_value=10, max_value=500, step=10)
                hp_units3 = hp.Int('units3', min_value=10, max_value=500, step=10)
                
                    
                model_RNN = Sequential()
                model_RNN.add(SimpleRNN(hp_units1, activation=hp_act_func, return_sequences=True, input_shape=(None, seq_size)))
                model_RNN.add(SimpleRNN(hp_units2, activation=hp_act_func))
                model_RNN.add(Dense(hp_units3, activation=hp_act_func))
                model_RNN.add(Dense(1, activation=hp_last_act_func))
                model_RNN.compile(optimizer=hp_optimizer, loss='mean_squared_error')
                
                return  model_RNN
            
            
            tuner = kt.Hyperband(model_builder,
                                objective='val_loss',
                                max_epochs=100,
                                factor=3,
                                directory='my_dir',
                                project_name='intro_to_kt')
            
            es_tuning = EarlyStopping(monitor='val_loss', 
                                patience=10, 
                                min_delta=0.001,
                                verbose=1,
                                restore_best_weights=True,
                                mode='min')
            
            tuner.search(trainX, trainY, epochs=500, validation_split=0.1, callbacks=[es_tuning])
            
            # Get the optimal hyperparameters
            best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
            
            model_RNN = tuner.hypermodel.build(best_hps)
            
            with open("myfile.txt", 'a') as f: 
                f.write('%s\n' % ('======================='))
                f.write('%s:%s\n' % ('model', 'RNN'))
                for key, value in best_hps.values.items(): 
                    f.write('%s:%s\n' % (key, value))
                    
            
            model_RNN.summary()
            print('Train...')
         
            es = EarlyStopping(monitor='val_loss', 
                                patience=20, 
                                min_delta=0.001,
                                verbose=2,
                                restore_best_weights=True,
                                mode='min')
            
            start_time = time.time()
            
            history = model_RNN.fit(trainX, trainY, epochs=run_epochs, 
                                #validation_split=0.1,
                                #batch_size=1,
                                verbose=2,
                                callbacks=[es],
                                validation_split=0.1
                                )
            
            comptime = (time.time() - start_time)
        
        else:
            print('=================RNN Model will run================================================')
            model_RNN = mk_ts.create_RNN_model(trainX, trainY, seq_size)
            model_RNN.summary()
            print('Train...')
            
            start_time = time.time()
            
            model_RNN.fit(trainX, trainY, validation_split=0.1,
                    verbose=2, epochs=run_epochs)
            
            comptime = (time.time() - start_time)
            
        # make predictions
        RNN_testPredict = model_RNN.predict(trainX[-test_size:])
        
        # calculate rmse, mse, mae, r2 error
        rmse, mse, mae, r2 = mk_ts.calc_perf_metrics(trainY[-test_size:], RNN_testPredict)
        bfattack_rmse_list.append(rmse)
        bfattack_mse_list.append(mse)
        bfattack_mae_list.append(mae)
        bfattack_comp_time.append(comptime)
        eps = 0

        result['bfattack_rmse_list'] = bfattack_rmse_list

        mk_ts.write_file("RNN", rmse, mse, mae, eps, comptime, cvs_file_name)
        
        

        if (run_FGSM == True):
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_RNN, "RNN", trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_RNN_after.pdf')
            
        if (run_AdvTrain == True):
            #adversarial traning
            print('=================run_AdvTrain================================================')
            #run_adversarial_traning(model_RNN,trainX[-test_size:], trainY[-test_size:])
            #end_idx = trainX.shape[0] - test_size
            #end_idx = 1000
            run_adversarial_traning(model_RNN,trainX[0:start_idx], trainY[0:start_idx])
                
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_RNN, "RNN", trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_RNN_after_advtrain.pdf')          
    
    #########################################################
    #LSTM  and Grid Search
    ######################################################
    if ((run_Models&run_LSTM) >= 1 ):
        if (run_GridSearch == True):
            print('===============LSTM with Grid Search Model will run================================================')
            comptime = 0
            ##########################
        
            def model_builder(hp):
                
                hp_activation = ['relu','sigmoid','softmax','softplus','softsign','tanh','selu', 'elu']
                hp_act_func = hp.Choice('act_func', values=hp_activation)
                hp_last_act_func = hp.Choice('last_act_func', values=hp_activation)
                
                optimizer_list = ['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam']
                hp_optimizer = hp.Choice('optimizer', values=optimizer_list)
                
                hp_units1 = hp.Int('units1', min_value=10, max_value=500, step=32)
                hp_units2 = hp.Int('units2', min_value=10, max_value=500, step=32)
                hp_units3 = hp.Int('units3', min_value=10, max_value=500, step=32)
                
            
                model= Sequential()
                model.add(LSTM(hp_units1, activation=hp_act_func, return_sequences=True, input_shape=(None, seq_size)))
                model.add(LSTM(hp_units2, activation=hp_act_func))
                model.add(Dense(hp_units3, activation=hp_act_func))
                model.add(Dense(1, activation=hp_last_act_func))
                model.compile(optimizer=hp_optimizer, loss='mean_squared_error')
                
                return  model
            
            tuner = kt.Hyperband(model_builder,
                                objective='val_loss',
                                max_epochs=100,
                                factor=3,
                                directory='my_dir',
                                project_name='intro_to_lstm')
            
            es_tuning = EarlyStopping(monitor='val_loss', 
                                patience=10, 
                                min_delta=0.001,
                                verbose=1,
                                restore_best_weights=True,
                                mode='min')
            
            tuner.search(trainX, trainY, epochs=500, validation_split=0.1, callbacks=[es_tuning])
            
            # Get the optimal hyperparameters
            best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
            
            model_LSTM = tuner.hypermodel.build(best_hps)
            
            with open("myfile.txt", 'a') as f: 
                f.write('%s\n' % ('======================='))
                f.write('%s:%s\n' % ('model', 'LSTM'))
                for key, value in best_hps.values.items(): 
                    f.write('%s:%s\n' % (key, value))
                
        
            model_LSTM.summary()
            print('Train...')
            
            es = EarlyStopping(monitor='val_loss', 
                                patience=20, 
                                min_delta=0.001,
                                verbose=2,
                                restore_best_weights=True,
                                mode='min')
            
            start_time = time.time()
            
            history = model_LSTM.fit(trainX, trainY, epochs=run_epochs, 
                                #validation_split=0.1,
                                #batch_size=1,
                                verbose=2,
                                callbacks=[es],
                                validation_split=0.1
                                )
            
            comptime = (time.time() - start_time)

        else:
            print('=================LSTM Model will run================================================')
            model_LSTM= mk_ts.create_LSTM_model(trainX, trainY, seq_size)
            model_LSTM.summary()
            print('Train...')
            
            start_time = time.time()
            
            model_LSTM.fit(trainX, trainY, validation_split=0.1,
                    verbose=1, epochs=run_epochs)
            
            comptime = (time.time() - start_time)
                
        # make predictions
        LSTM_testPredict = model_LSTM.predict(trainX[-test_size:])
        
        # calculate rmse, mse, mae, r2 error
        rmse, mse, mae, r2 = mk_ts.calc_perf_metrics(trainY[-test_size:], LSTM_testPredict)
        bfattack_rmse_list.append(rmse)
        bfattack_mse_list.append(mse)
        bfattack_mae_list.append(mae)
        bfattack_comp_time.append(comptime)
        eps = 0
        mk_ts.write_file("LSTM", rmse, mse, mae, eps, comptime, cvs_file_name)
        
        
        if (run_FGSM == True):
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_LSTM,"LSTM", trainX[-test_size:], trainY[-test_size:],  cvs_file_name, 'attack_LSTM_after.pdf')
            
        if (run_AdvTrain == True):
            #adversarial traning
            print('=================run_AdvTrain================================================')
            run_adversarial_traning(model_LSTM, trainX[0:start_idx], trainY[0:start_idx])
                
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_LSTM, "LSTM",  trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_LSTM_after_advtrain.pdf')     


    #########################################################
    #BiLSTM  and Grid Search
    ######################################################
    if ((run_Models&run_BiLSTM) >= 1 ):
        if (run_GridSearch == True):
            print('===============run_BiLSTM with Grid Search Model will run================================================')
            comptime = 0
            ##########################

            def model_builder(hp):
                
                hp_activation = ['relu','sigmoid','softmax','softplus','softsign','tanh','selu', 'elu']
                hp_act_func = hp.Choice('act_func', values=hp_activation)
                hp_last_act_func = hp.Choice('last_act_func', values=hp_activation)
                
                optimizer_list = ['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam']
                hp_optimizer = hp.Choice('optimizer', values=optimizer_list)
                
                hp_units1 = hp.Int('units1', min_value=10, max_value=500, step=32)
                hp_units2 = hp.Int('units2', min_value=10, max_value=500, step=32)
                hp_units3 = hp.Int('units3', min_value=10, max_value=500, step=32)
                
            
                model= Sequential()
                model.add(Bidirectional(LSTM(hp_units1, activation=hp_act_func), input_shape=(None, seq_size)))
                model.add(Dense(1, activation=hp_last_act_func))
                model.compile(optimizer=hp_optimizer, loss='mean_squared_error')
                
                
                return  model
            
            tuner = kt.Hyperband(model_builder,
                                objective='val_loss',
                                max_epochs=100,
                                factor=3,
                                directory='my_dir',
                                project_name='intro_to_bilstm')
            
            es_tuning = EarlyStopping(monitor='val_loss', 
                                patience=10, 
                                min_delta=0.001,
                                verbose=1,
                                restore_best_weights=True,
                                mode='min')
            
            tuner.search(trainX, trainY, epochs=500, validation_split=0.1, callbacks=[es_tuning])
            
            # Get the optimal hyperparameters
            best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
            
            model_BiLSTM= tuner.hypermodel.build(best_hps)
            
            with open("myfile.txt", 'a') as f: 
                f.write('%s\n' % ('======================='))
                f.write('%s:%s\n' % ('model', 'BiLSTM'))
                for key, value in best_hps.values.items(): 
                    f.write('%s:%s\n' % (key, value))
                    
            
            model_BiLSTM.summary()
            print('Train...')
            
            es = EarlyStopping(monitor='val_loss', 
                                patience=20, 
                                min_delta=0.001,
                                verbose=2,
                                restore_best_weights=True,
                                mode='min')
            
            start_time = time.time()
            
            history = model_BiLSTM.fit(trainX, trainY, epochs=run_epochs, 
                                #validation_split=0.1,
                                #batch_size=1,
                                verbose=2,
                                callbacks=[es],
                                validation_split=0.1
                                )
            
            comptime = (time.time() - start_time)

        else:
            print('BiLSTM Model will run================================================')
            ###############################################
            #Bidirectional LSTM (BiLSTM)
            ###############################################
            model_BiLSTM= mk_ts.create_BiLSTM_model(trainX, trainY, seq_size)
            model_BiLSTM.summary()
            print('Train...')
            
            start_time = time.time()
            
            model_BiLSTM.fit(trainX, trainY, validation_split=0.1,
                    verbose=1, epochs=run_epochs)
            
            comptime = (time.time() - start_time)
                
        # make predictions
        BiLSTM_testPredict = model_BiLSTM.predict(trainX[-test_size:])
        
        # calculate rmse, mse, mae, r2 error
        rmse, mse, mae, r2 = mk_ts.calc_perf_metrics(trainY[-test_size:], BiLSTM_testPredict)
        bfattack_rmse_list.append(rmse)
        bfattack_mse_list.append(mse)
        bfattack_mae_list.append(mae)
        bfattack_comp_time.append(comptime)
        eps = 0
        mk_ts.write_file("BiLSTM", rmse, mse, mae, eps, comptime, cvs_file_name)
        
        
        if (run_FGSM == True):
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_BiLSTM, "BiLSTM",  trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_BiLSTM_after.pdf')
            
        if (run_AdvTrain == True):
            #adversarial traning
            print('=================run_AdvTrain================================================')
            run_adversarial_traning(model_BiLSTM, trainX[0:start_idx], trainY[0:start_idx])
                
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_BiLSTM, "BiLSTM",  trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_BiLSTM_after_advtrain.pdf')  
            
            
    #########################################################
    #GRU  and Grid Search
    ######################################################
    if ((run_Models&run_GRU) >= 1 ):
        if (run_GridSearch == True):
            print('===============run_BiLSTM with Grid Search Model will run================================================')
            comptime = 0
            ##########################

            def model_builder(hp):
                
                hp_activation = ['relu','sigmoid','softmax','softplus','softsign','tanh','selu', 'elu']
                hp_act_func = hp.Choice('act_func', values=hp_activation)
                hp_last_act_func = hp.Choice('last_act_func', values=hp_activation)
                
                optimizer_list = ['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam']
                hp_optimizer = hp.Choice('optimizer', values=optimizer_list)
                
                hp_units1 = hp.Int('units1', min_value=10, max_value=500, step=32)
                hp_units2 = hp.Int('units2', min_value=10, max_value=500, step=32)
                hp_units3 = hp.Int('units3', min_value=10, max_value=500, step=32)
                
            
                model= Sequential()
                model.add(GRU(hp_units1, activation=hp_act_func, return_sequences=True, input_shape=(None, seq_size)))
                model.add(GRU(hp_units2, activation=hp_act_func))
                model.add(Dense(hp_units3, activation=hp_act_func))
                model.add(Dense(1, activation=hp_last_act_func))
                model.compile(optimizer=hp_optimizer, loss='mean_squared_error')  
                
                return  model
            
            tuner = kt.Hyperband(model_builder,
                                objective='val_loss',
                                max_epochs=100,
                                factor=3,
                                directory='my_dir',
                                project_name='intro_to_gru')
            
            es_tuning = EarlyStopping(monitor='val_loss', 
                                patience=10, 
                                min_delta=0.001,
                                verbose=1,
                                restore_best_weights=True,
                                mode='min')
            
            tuner.search(trainX, trainY, epochs=500, validation_split=0.1, callbacks=[es_tuning])
            
            # Get the optimal hyperparameters
            best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
            
            model_GRU = tuner.hypermodel.build(best_hps)
            
            with open("myfile.txt", 'a') as f: 
                f.write('%s\n' % ('======================='))
                f.write('%s:%s\n' % ('model', 'GRU'))
                for key, value in best_hps.values.items(): 
                    f.write('%s:%s\n' % (key, value))
                    
            
            model_GRU.summary()
            print('Train...')
            
            es = EarlyStopping(monitor='val_loss', 
                                patience=20, 
                                min_delta=0.001,
                                verbose=2,
                                restore_best_weights=True,
                                mode='min')
            
            start_time = time.time()
            
            history = model_GRU.fit(trainX, trainY, epochs=run_epochs, 
                                #validation_split=0.1,
                                #batch_size=1,
                                verbose=2,
                                callbacks=[es],
                                validation_split=0.1
                                )
            
            comptime = (time.time() - start_time)

        else:
            print('GRU Model will run================================================')
            #########################################################
            #GRU
            ######################################################
            model_GRU = mk_ts.create_GRU_model(trainX, trainY, seq_size)
            model_GRU.summary()
            print('Train...')
            
            start_time = time.time()
            model_GRU.fit(trainX, trainY, validation_split=0.1,
                    verbose=2, epochs=run_epochs)
            comptime = (time.time() - start_time)
        
        # make predictions
        GRU_testPredict = model_GRU.predict(trainX[-test_size:])
        
        # calculate rmse, mse, mae, r2 error
        rmse, mse, mae, r2 = mk_ts.calc_perf_metrics(trainY[-test_size:], GRU_testPredict)
        bfattack_rmse_list.append(rmse)
        bfattack_mse_list.append(mse)
        bfattack_mae_list.append(mae)
        bfattack_comp_time.append(comptime)
        eps = 0
        mk_ts.write_file("GRU", rmse, mse, mae, eps, comptime, cvs_file_name)
        
        if (run_FGSM == True):
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_GRU, "GRU",  trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_GRU_after.pdf')
        
        if (run_AdvTrain == True):
            #adversarial traning
            print('=================run_AdvTrain================================================')
            run_adversarial_traning(model_GRU, trainX[0:start_idx], trainY[0:start_idx])
                
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_GRU, "GRU",  trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_GRU_after_advtrain.pdf')  
            
            
    #########################################################
    #LSTM_Attention  and Grid Search
    ######################################################
    if ((run_Models&run_LSTM_Attention) >= 1 ):
        if (run_GridSearch == True):
            print('===============LSTM_Attention with Grid Search Model will run================================================')
            comptime = 0
            ##########################
        
            def model_builder(hp):
                
                hp_activation = ['relu','sigmoid','softmax','softplus','softsign','tanh','selu', 'elu']
                hp_act_func = hp.Choice('act_func', values=hp_activation)
                hp_last_act_func = hp.Choice('last_act_func', values=hp_activation)
                
                optimizer_list = ['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam']
                hp_optimizer = hp.Choice('optimizer', values=optimizer_list)
                
                hp_units1 = hp.Int('units1', min_value=10, max_value=500, step=32)
                hp_units2 = hp.Int('units2', min_value=10, max_value=500, step=32)
                hp_units3 = hp.Int('units3', min_value=10, max_value=500, step=32)
                
            
                # model= Sequential()
                # model.add(LSTM(hp_units1, activation=hp_act_func, return_sequences=True, input_shape=(None, seq_size)))
                # model.add(LSTM(hp_units2, activation=hp_act_func))
                # model.add(Dense(hp_units3, activation=hp_act_func))
                # model.add(Dense(1, activation=hp_last_act_func))
                # model.compile(optimizer=hp_optimizer, loss='mean_squared_error')
                
                model = Sequential()
                model.add(LSTM(hp_units1, activation=hp_act_func, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
                #model.add(LSTM(32, activation='relu', return_sequences=False))
                model.add(Attention(return_sequences=False)) # receive 3D and output 3D
                model.add(Dropout(0.2))
                model.add(Dense(trainY.shape[1],activation=hp_last_act_func))
                model.compile(optimizer=hp_optimizer, loss='mse')
                
                return  model
            
            tuner = kt.Hyperband(model_builder,
                                objective='val_loss',
                                max_epochs=100,
                                factor=3,
                                directory='my_dir',
                                project_name='intro_to_lstm_attention')
            
            es_tuning = EarlyStopping(monitor='val_loss', 
                                patience=10, 
                                min_delta=0.001,
                                verbose=1,
                                restore_best_weights=True,
                                mode='min')
            
            tuner.search(trainX, trainY, epochs=500, validation_split=0.1, callbacks=[es_tuning])
            
            # Get the optimal hyperparameters
            best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
            
            model_LSTM_Attention = tuner.hypermodel.build(best_hps)
            
            with open("myfile.txt", 'a') as f: 
                f.write('%s\n' % ('======================='))
                f.write('%s:%s\n' % ('model', 'LSTM_Attention'))
                for key, value in best_hps.values.items(): 
                    f.write('%s:%s\n' % (key, value))
                
        
            model_LSTM_Attention.summary()
            print('Train...')
            
            es = EarlyStopping(monitor='val_loss', 
                                patience=20, 
                                min_delta=0.001,
                                verbose=2,
                                restore_best_weights=True,
                                mode='min')
            
            start_time = time.time()
            
            history = model_LSTM_Attention.fit(trainX, trainY, epochs=run_epochs, 
                                #validation_split=0.1,
                                #batch_size=1,
                                verbose=2,
                                callbacks=[es],
                                validation_split=0.1
                                )
            
            comptime = (time.time() - start_time)

        else:
            print('=================LSTM_Attention Model will run================================================')
            model_LSTM_Attention= mk_ts.create_LSTM_Attention_model(trainX, trainY, seq_size)
            model_LSTM_Attention.summary()
            print('Train...')
            
            start_time = time.time()
            
            model_LSTM_Attention.fit(trainX, trainY, epochs=run_epochs, batch_size=run_batch_size, validation_split=0.1, verbose=1)
            
            comptime = (time.time() - start_time)
                
        # make predictions
        LSTM__Attention_testPredict = model_LSTM_Attention.predict(trainX[-test_size:])
        prediction_copies = np.repeat(LSTM__Attention_testPredict, df_for_training.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
        
        # calculate rmse, mse, mae, r2 error
        rmse, mse, mae, r2 = mk_ts.calc_perf_metrics(trainY[-test_size:], y_pred_future)
        bfattack_rmse_list.append(rmse)
        bfattack_mse_list.append(mse)
        bfattack_mae_list.append(mae)
        bfattack_comp_time.append(comptime)
        eps = 0
        mk_ts.write_file("LSTM_Attention", rmse, mse, mae, eps, comptime, cvs_file_name)
        
        
        if (run_FGSM == True):
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_LSTM_Attention, "LSTM_Attention", trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_LSTM_Attention_after.pdf')
            
        if (run_AdvTrain == True):
            #adversarial traning
            print('=================run_AdvTrain================================================')
            run_adversarial_traning(model_LSTM_Attention, trainX[0:start_idx], trainY[0:start_idx])
                
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_LSTM_Attention, "LSTM_Attention", trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_LSTM_Attention_after_advtrain.pdf')  
            

    #########################################################
    #BiLSTM_Attention  and Grid Search
    ###################################################### 
    if ((run_Models&run_BiLSTM_Attention) >= 1 ):
        if (run_GridSearch == True):
            print('===============LSTM with Grid Search Model will run================================================')
            comptime = 0
            ##########################
        
            def model_builder(hp):
                
                hp_activation = ['relu','sigmoid','softmax','softplus','softsign','tanh','selu', 'elu']
                hp_act_func = hp.Choice('act_func', values=hp_activation)
                hp_last_act_func = hp.Choice('last_act_func', values=hp_activation)
                
                optimizer_list = ['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam']
                hp_optimizer = hp.Choice('optimizer', values=optimizer_list)
                
                hp_units1 = hp.Int('units1', min_value=10, max_value=500, step=32)
                hp_units2 = hp.Int('units2', min_value=10, max_value=500, step=32)
                hp_units3 = hp.Int('units3', min_value=10, max_value=500, step=32)
                
            
                # model= Sequential()
                # model.add(LSTM(hp_units1, activation=hp_act_func, return_sequences=True, input_shape=(None, seq_size)))
                # model.add(LSTM(hp_units2, activation=hp_act_func))
                # model.add(Dense(hp_units3, activation=hp_act_func))
                # model.add(Dense(1, activation=hp_last_act_func))
                # model.compile(optimizer=hp_optimizer, loss='mean_squared_error')
                
                model = Sequential()
                model.add(Bidirectional(LSTM(hp_units1, activation=hp_act_func, return_sequences=True), input_shape=(trainX.shape[1], trainX.shape[2])))
                #model.add(Bidirectional(LSTM(64, activation='relu')))
                model.add(Attention(return_sequences=False)) # receive 3D and output 3D
                #model.add(Bidirectional(LSTM(100, activation='relu')))
                model.add(Dropout(0.2))
                model.add(Dense(1, activation=hp_act_func))
                model.compile(optimizer=hp_optimizer, loss='mean_squared_error')
                
                return  model
            
            tuner = kt.Hyperband(model_builder,
                                objective='val_loss',
                                max_epochs=100,
                                factor=3,
                                directory='my_dir',
                                project_name='intro_to_bilstm_attention')
            
            es_tuning = EarlyStopping(monitor='val_loss', 
                                patience=10, 
                                min_delta=0.001,
                                verbose=1,
                                restore_best_weights=True,
                                mode='min')
            
            tuner.search(trainX, trainY, epochs=500, validation_split=0.1, callbacks=[es_tuning])
            
            # Get the optimal hyperparameters
            best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
            
            model_BiLSTM_Attention = tuner.hypermodel.build(best_hps)
            
            with open("myfile.txt", 'a') as f: 
                f.write('%s\n' % ('======================='))
                f.write('%s:%s\n' % ('model', 'BiLSTM_Attention'))
                for key, value in best_hps.values.items(): 
                    f.write('%s:%s\n' % (key, value))
                
        
            model_BiLSTM_Attention.summary()
            print('Train...')
            
            es = EarlyStopping(monitor='val_loss', 
                                patience=20, 
                                min_delta=0.001,
                                verbose=2,
                                restore_best_weights=True,
                                mode='min')
            
            start_time = time.time()
            
            history = model_BiLSTM_Attention.fit(trainX, trainY, epochs=run_epochs, 
                                #validation_split=0.1,
                                #batch_size=1,
                                verbose=2,
                                callbacks=[es],
                                validation_split=0.1
                                )
            
            comptime = (time.time() - start_time)

        else:
            print('=================BiLSTM_Attention Model will run================================================')
            model_BiLSTM_Attention= mk_ts.create_BiLSTM_Attention_model(trainX, trainY, seq_size)
            model_BiLSTM_Attention.summary()
            print('Train...')
            
            start_time = time.time()
            
            model_BiLSTM_Attention.fit(trainX, trainY, epochs=run_epochs, batch_size=run_batch_size, validation_split=0.1, verbose=1)
            
            comptime = (time.time() - start_time)
                
        # make predictions
        BiLSTM__Attention_testPredict = model_BiLSTM_Attention.predict(trainX[-test_size:])
        prediction_copies = np.repeat(BiLSTM__Attention_testPredict, df_for_training.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
        
        # calculate rmse, mse, mae, r2 error
        rmse, mse, mae, r2 = mk_ts.calc_perf_metrics(trainY[-test_size:], y_pred_future)
        bfattack_rmse_list.append(rmse)
        bfattack_mse_list.append(mse)
        bfattack_mae_list.append(mae)
        bfattack_comp_time.append(comptime)
        eps = 0
        mk_ts.write_file("BiLSTM_Attention", rmse, mse, mae, eps, comptime, cvs_file_name)
        
        
        if (run_FGSM == True):
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_BiLSTM_Attention, "BiLSTM_Attention", trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_BiLSTM_Attention_after.pdf')
            
        if (run_AdvTrain == True):
            #adversarial traning
            print('=================run_AdvTrain================================================')
            run_adversarial_traning(model_BiLSTM_Attention, trainX[0:start_idx], trainY[0:start_idx])
                
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_BiLSTM_Attention, "BiLSTM_Attention", trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_BiLSTM_Attention_after_advtrain.pdf')     
            
            
    #########################################################
    #LSTM_Encoder  and Grid Search
    ###################################################### 
    if ((run_Models&run_LSTM_Encoder) >= 1 ):
        if (run_GridSearch == True):
            print('===============LSTM Encoder with Grid Search Model will run================================================')
            comptime = 0
            ##########################
        
            def model_builder(hp):
                
                hp_activation = ['relu','sigmoid','softmax','softplus','softsign','tanh','selu', 'elu']
                hp_act_func = hp.Choice('act_func', values=hp_activation)
                hp_last_act_func = hp.Choice('last_act_func', values=hp_activation)
                
                optimizer_list = ['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam']
                hp_optimizer = hp.Choice('optimizer', values=optimizer_list)
                
                hp_units1 = hp.Int('units1', min_value=10, max_value=500, step=32)
                hp_units2 = hp.Int('units2', min_value=10, max_value=500, step=32)
                hp_units3 = hp.Int('units3', min_value=10, max_value=500, step=32)
                
            
                # model= Sequential()
                # model.add(LSTM(hp_units1, activation=hp_act_func, return_sequences=True, input_shape=(None, seq_size)))
                # model.add(LSTM(hp_units2, activation=hp_act_func))
                # model.add(Dense(hp_units3, activation=hp_act_func))
                # model.add(Dense(1, activation=hp_last_act_func))
                # model.compile(optimizer=hp_optimizer, loss='mean_squared_error')
                
                model = Sequential()
                model.add(LSTM(hp_units1, activation=hp_act_func, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
                model.add(LSTM(hp_units2, activation=hp_act_func, return_sequences=False))
                #model.add(Dropout(rate=0.2))
                #model.add(RepeatVector(trainX.shape[1]))
                model.add(RepeatVector(trainY.shape[1]))
                model.add(LSTM(hp_units3, activation=hp_act_func, return_sequences=True))
                model.add(LSTM(hp_units3, activation=hp_act_func, return_sequences=False))
                #model.add(Dropout(rate=0.2))
                #model.add(TimeDistributed(Dense(trainY.shape[2])))
                model.add(Dense(trainY.shape[1],activation=hp_act_func))
                model.compile(optimizer=hp_optimizer, loss='mse')
                
                return  model
            
            tuner = kt.Hyperband(model_builder,
                                objective='val_loss',
                                max_epochs=100,
                                factor=3,
                                directory='my_dir',
                                project_name='intro_to_lstm_encoder')
            
            es_tuning = EarlyStopping(monitor='val_loss', 
                                patience=10, 
                                min_delta=0.001,
                                verbose=1,
                                restore_best_weights=True,
                                mode='min')
            
            tuner.search(trainX, trainY, epochs=500, validation_split=0.1, callbacks=[es_tuning])
            
            # Get the optimal hyperparameters
            best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
            
            model_LSTM_Encoder = tuner.hypermodel.build(best_hps)
            
            with open("myfile.txt", 'a') as f: 
                f.write('%s\n' % ('======================='))
                f.write('%s:%s\n' % ('model', 'LSTM_Encoder'))
                for key, value in best_hps.values.items(): 
                    f.write('%s:%s\n' % (key, value))
                
        
            model_LSTM_Encoder.summary()
            print('Train...')
            
            es = EarlyStopping(monitor='val_loss', 
                                patience=20, 
                                min_delta=0.001,
                                verbose=2,
                                restore_best_weights=True,
                                mode='min')
            
            start_time = time.time()
            
            history = model_LSTM_Encoder.fit(trainX, trainY, epochs=run_epochs, 
                                #validation_split=0.1,
                                #batch_size=1,
                                verbose=2,
                                callbacks=[es],
                                validation_split=0.1
                                )
            
            comptime = (time.time() - start_time)

        else:
            print('=================LSTM_Encoder Model will run================================================')
            model_LSTM_Encoder= mk_ts.create_LSTM_Encoder_model(trainX, trainY, seq_size)
            model_LSTM_Encoder.summary()
            print('Train...')
            
            start_time = time.time()
            
            model_LSTM_Encoder.fit(trainX, trainY, epochs=run_epochs, batch_size=run_batch_size, validation_split=0.1, verbose=1)
            
            comptime = (time.time() - start_time)
                
        # make predictions
        LSTM_Encoder_testPredict = model_LSTM_Encoder.predict(trainX[-test_size:])
        #LSTM_Encoder_testPredict = LSTM_Encoder_testPredict[:,0]
        prediction_copies = np.repeat(LSTM_Encoder_testPredict, df_for_training.shape[1], axis=-1)
        #y_pred_future = scaler.inverse_transform(LSTM_Encoder_testPredict)
        y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
        
        # calculate rmse, mse, mae, r2 error
        rmse, mse, mae, r2 = mk_ts.calc_perf_metrics(trainY[-test_size:], y_pred_future)
        bfattack_rmse_list.append(rmse)
        bfattack_mse_list.append(mse)
        bfattack_mae_list.append(mae)
        bfattack_comp_time.append(comptime)
        eps = 0
        mk_ts.write_file("LSTM_Encoder", rmse, mse, mae, eps, comptime, cvs_file_name)
        
        
        if (run_FGSM == True):
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_LSTM_Encoder, "LSTM_Encoder", trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_LSTM_Encoder_after.pdf')
            
        if (run_AdvTrain == True):
            #adversarial traning
            print('=================run_AdvTrain================================================')
            run_adversarial_traning(model_LSTM_Encoder,trainX[0:start_idx], trainY[0:start_idx])
                
            #after attack
            print('=================run_FGSM================================================')
            predict_after_attack(model_LSTM_Encoder, "LSTM_Encoder", trainX[-test_size:], trainY[-test_size:], cvs_file_name, 'attack_LSTM_Encoder_after_advtrain.pdf')
            
    ###############################################
    ###############################################
    file_name = 'before_attack.pdf'
    #draw_before_attack(trainY[-test_size:], BiLSTM__Attention_testPredict, BiLSTM__Attention_testPredict, BiLSTM__Attention_testPredict, BiLSTM__Attention_testPredict, BiLSTM__Attention_testPredict, BiLSTM__Attention_testPredict, file_name)

    #Draw all results
    ###############################################
    ###############################################
    file_name = 'before_attack.pdf'
    draw_before_attack(trainY[-test_size:], RNN_testPredict, LSTM_testPredict, BiLSTM_testPredict, GRU_testPredict, LSTM__Attention_testPredict, BiLSTM__Attention_testPredict, file_name)

    return result
