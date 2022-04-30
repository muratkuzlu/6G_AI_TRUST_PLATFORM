#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:39:40 2021

@author: ozgur
"""

import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tqdm.notebook import tqdm
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

import timeseries_functions as mk_ts

DS_PATH = '/Users/ozgur/Documents/Workshop/DeepMIMO_Dataset_Generation_v1.1/DLCB_dataset/'

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data


        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

def get_model(n_beams, act_func = 'relu', input_dim=256,
              loss_fn='mean_squared_error', multiplication_factor=1.0):
    model = Sequential()
    model.add(Dense(int(100*multiplication_factor), input_dim=input_dim, activation=act_func))
    model.add(Dense(int(100*multiplication_factor), activation=act_func))
    model.add(Dense(int(100*multiplication_factor), activation=act_func))
    model.add(Dense(int(100*multiplication_factor), activation=act_func))
    model.add(Dense(n_beams, activation=act_func))
    model.compile(loss=loss_fn, optimizer='rmsprop', 
                  metrics=['mean_squared_error'])
    
    return model


def get_dataset(sc_name):
    In_set_file=loadmat(DS_PATH + sc_name + '_DLCB_input.mat')
    Out_set_file=loadmat(DS_PATH + sc_name + '_DLCB_output.mat')
    
    In_set=In_set_file['DL_input']
    Out_set=Out_set_file['DL_output']
    
    In_set_real = np.zeros((In_set.shape[0], In_set.shape[1]*2))
    for i in range(In_set.shape[0]):
        for j in range(In_set.shape[1]):
            In_set_real[i,j*2] = np.real(In_set[i,j])
            In_set_real[i,j*2+1] = np.imag(In_set[i,j])
            
    num_user_tot=In_set.shape[0]
    
    In_train, In_test, Out_train, Out_test = train_test_split(In_set_real, Out_set, test_size=0.2)
    return In_train, In_test, Out_train, Out_test

# Model training function
def train(In_train, Out_train, In_test, Out_test,
          nb_epoch, batch_size,
          loss_fn,n_BS,n_beams, sc_name):

    AP_models = []
    AP_models_history = []
    for idx in tqdm(range(0, n_BS*n_beams-2, n_beams)):
        idx_str = str(idx / n_beams + 1)
        model = get_model(n_beams, loss_fn=loss_fn, input_dim=n_beams)
        mcp_save = ModelCheckpoint('models/' + sc_name + '_bs_' + str(int(np.float(idx_str)-1.0))+ '.hdf5', 
                                   save_best_only=True, verbose=0, 
                                   monitor='val_mean_squared_error', mode='min')
        
        es = EarlyStopping(monitor='val_loss', 
                           patience=20, 
                            #min_delta=0.0001,
                           verbose=1,
                           restore_best_weights=True,
                           mode='min')
        
        history = model.fit(In_train,
                            Out_train[:, idx:idx + n_beams],
                            batch_size=batch_size,
                            epochs=nb_epoch,
                            verbose=0,
                            validation_data=(In_test, Out_test[:,idx:idx + n_beams]),
                            callbacks=[mcp_save,es])
        
        AP_models_history.append(history)
        
        AP_models.append(model)
    return AP_models, AP_models_history

def load_models(sc_name, n_BS, model_ext=''):
    AP_models = []
    for i in range(n_BS):
        model_path = 'models/' + sc_name + '_bs_' + str(i) + model_ext + '.hdf5'
        AP_models.append(keras.models.load_model(model_path))
    return AP_models

def attack_models(model, attack_name, eps_val, testset, outout_labels, norm=np.inf):
    #logger.debug("Attack started" + attack_name)
    logits_model = tf.keras.Model(model.input, model.layers[-1].output)
    if attack_name == 'FGSM':
        '''
        In_test_adv = fast_gradient_method(logits_model, testset, eps_val,
                                           norm, targeted=False,
                                           clip_min=testset.min(),clip_max=testset.max())
        '''
        #print('111')
        In_test_adv = [mk_ts.my_fgsm([x_input,],[y_output,],model,eps_val)[0].numpy() for x_input, y_output in zip(testset, outout_labels)]
        #print('222')
        return np.array(In_test_adv)
    elif attack_name == 'PGD':
        In_test_adv = projected_gradient_descent(logits_model, testset, eps=eps_val, 
                                         norm=norm,nb_iter=50,eps_iter=eps_val/10.0,
                                         targeted=False,
                                         clip_min=testset.min(),clip_max=testset.max())
        return In_test_adv
    elif attack_name == 'BIM':
        In_test_adv = basic_iterative_method(logits_model,testset,eps_val,
                                             eps_iter=eps_val/10.0, nb_iter=50,norm=norm,
                                             targeted=False)
        return In_test_adv
    elif attack_name == 'MIM':
        In_test_adv = momentum_iterative_method(logits_model,testset,eps_val,
                                             eps_iter=eps_val/10.0, nb_iter=50,norm=norm,
                                             targeted=False,
                                             clip_min=testset.min(),clip_max=testset.max())
        return In_test_adv
    elif attack_name == 'CW':
        In_test_adv = []
        for i in tqdm(range(100)):
            tmp = carlini_wagner_l2(logits_model, testset[i:i+1,:].astype(np.float32),
                                    targeted=False,# y=[np.float64(0.0)],
                                    batch_size=512, confidence=100.0,
                                    abort_early=False, max_iterations=100,
                                    clip_min=testset.min(),clip_max=testset.max())
            In_test_adv.append(tmp)
        
        return np.array(In_test_adv)
    elif attack_name == 'MIM':
        print('*'*20)
        In_test_adv = momentum_iterative_method(logits_model,testset,
                                                eps=eps_val,eps_iter=0.2,
                                                nb_iter=50,norm=norm)
        return In_test_adv
    
def adv_train(In_train, Out_train, In_test, Out_test,
          nb_epoch, batch_size,
          loss_fn,n_BS,n_beams,
          attack_types, eps_val, sc_name):
    AP_models = []
    for idx in tqdm(range(0, n_BS*n_beams-2, n_beams), position=0, leave=True):
        idx_str = str(idx / n_beams + 1)
        model = get_model(n_beams, loss_fn=loss_fn, input_dim=n_beams)
        
        model_f_name = 'models/' + sc_name + '_bs_' + str(int(np.float(idx_str)-1.0)) + '_' + str(eps_val) + '_adv.hdf5'
        mcp_save = ModelCheckpoint(model_f_name, 
                                   save_best_only=True, verbose=0, 
                                   monitor='val_mean_squared_error', mode='min')
        
        es = EarlyStopping(monitor='val_loss', 
                           patience=10, 
                            #min_delta=0.0001,
                           verbose=0,
                           restore_best_weights=True,
                           mode='min')
        
        history = model.fit(In_train,
                            Out_train[:, idx:idx + n_beams],
                            batch_size=batch_size,
                            epochs=nb_epoch,
                            verbose=0,
                            validation_data=(In_test, Out_test[:,idx:idx + n_beams]),
                            callbacks=[mcp_save,es])
        
        for attack in attack_types:
            In_train_adv = attack_models(model, attack_name=attack, eps_val=eps_val, testset=In_train)
            
            In_train_adv = np.concatenate((In_train, In_train_adv), axis=0)
            Out_train_adv = np.concatenate((Out_train, Out_train), axis=0)
            
            for i in range(5):
                es_adv = EarlyStopping(monitor='val_loss', 
                                   patience=10, 
                                    #min_delta=0.0001,
                                   verbose=0,
                                   restore_best_weights=True,
                                   mode='min')
                model.fit(In_train_adv,
                            Out_train_adv[:, idx:idx + n_beams],
                            batch_size=batch_size,
                            epochs=nb_epoch,
                            verbose=0,
                            validation_data=(In_test, Out_test[:,idx:idx + n_beams]),
                            callbacks=[mcp_save,es_adv])
        AP_models.append(model)
    return AP_models

# Model training function
def train_distill(In_train, Out_train, In_test, Out_test,
          nb_epoch, batch_size,
          loss_fn,n_BS,n_beams, student_multiplication, 
          teacher_multiplication, sc_name, distill_alpha, distill_temp):

    AP_models = []
    for idx in tqdm(range(0, n_BS*n_beams-2, n_beams)):
        idx_str = str(idx / n_beams + 1)
        
        student_loss_fn = keras.losses.MeanSquaredError()
        teacher_model = get_model(n_beams, loss_fn=loss_fn, 
                                  input_dim=n_beams,multiplication_factor=teacher_multiplication)
        student_model = get_model(n_beams, loss_fn=loss_fn,
                                  input_dim=n_beams,multiplication_factor=student_multiplication)
        
        #student_model_scratch = keras.models.clone_model(student_model)
        
        teacher_model.compile(optimizer='rmsprop',
                              #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              loss=keras.losses.MeanSquaredError(),
                              metrics=['mean_squared_error'],
                              )

        
        model_save_name = 'models/' + sc_name + '_bs_' + str(int(np.float(idx_str)-1.0))+ '_distilled.hdf5'
        
        es_teacher = EarlyStopping(monitor='val_loss', 
                                   patience=20, 
                                   #min_delta=0.0001,
                                   verbose=0,
                                   restore_best_weights=True,
                                   mode='min')
                        
        teacher_history = teacher_model.fit(In_train,
                                            Out_train[:, idx:idx + n_beams],
                                            batch_size=batch_size,
                                            epochs=nb_epoch,
                                            verbose=0,
                                            callbacks=[es_teacher],
                                            validation_data=(In_test, Out_test[:,idx:idx + n_beams]))
        
        distiller = Distiller(student=student_model, teacher=teacher_model)
        distiller.compile(optimizer='adam',
                            metrics=['mean_squared_error'],
                            student_loss_fn=student_loss_fn,
                            distillation_loss_fn=keras.losses.KLDivergence(),
                            alpha=distill_alpha,
                            temperature=distill_alpha,
                        )
        
        es = EarlyStopping(monitor='student_loss', 
                           patience=20, 
                           #min_delta=0.0001,
                           verbose=0,
                           restore_best_weights=True,
                           mode='min')
        
        distiller.fit(In_train,
                     Out_train[:, idx:idx + n_beams],
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        verbose=0,
                        callbacks=[es],
                        validation_data=(In_test, Out_test[:,idx:idx + n_beams]))
        
        distiller.student.save(model_save_name)
        
        AP_models.append(distiller.student)
    return AP_models