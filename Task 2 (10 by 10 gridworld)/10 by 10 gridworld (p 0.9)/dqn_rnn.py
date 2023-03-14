# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:08:48 2021

@author: 
"""
from __future__ import absolute_import
from __future__ import print_function
#from sumolib import checkBinary

import os
import sys
import optparse
import subprocess
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
from collections import deque
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, LSTM, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


class DRQNAgent:
    def __init__(self, state_size, action_size, state_sequence_size, label_sequence_size, num_label):
        self.state_size = state_size
        self.action_size = action_size
        self.state_sequence_size = state_sequence_size
        self.label_sequence_size = label_sequence_size
        self.num_label = num_label
        self.memory = deque(maxlen=int(1e6)) # =2000
        #self.memory = []
        self.all_memory = []
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001 #0.01
        self.epsilon_decay = 0.998 #0.995
        self.learning_rate = 0.001
        self.eval_model = self._build_model()
        self.tar_model = self._build_model()
        
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        
        input_state = Input(shape=(self.state_sequence_size, self.state_size,)) # Variable-length sequence of ints, states
        input_label = Input(shape=(self.label_sequence_size, self.num_label,)) # Variable-length sequence of ints, labels
        
        state_features = LSTM(self.state_size, return_sequences=False)(input_state)
        label_features = LSTM(self.num_label, return_sequences=False)(input_label)
        
        x = concatenate([state_features, label_features])
        flatten_x = Flatten()(x)
        
        dense_1 = Dense(32, activation=None)(flatten_x)
        dense_2 = Dense(16, activation=None)(dense_1)
        dense_out = Dense(self.action_size, activation='relu')(dense_2) #kernel_initializer=initializers.RandomNormal(stddev=0.01),    bias_initializer=initializers.Zeros()
        
        model = keras.Model(
            inputs=[input_state, input_label],
            outputs=[dense_out],)
        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def memorize(self, state_sequence, label_sequence, action, reward, next_state_sequence, next_label_sequence):
        state_sequence_mem = state_sequence.copy()
        label_sequence_mem = label_sequence.copy()
        action_mem = action
        reward_mem = reward
        next_state_sequence_mem = next_state_sequence.copy()
        next_label_sequence_mem = next_label_sequence.copy()
        self.memory.append((state_sequence_mem, label_sequence_mem, action_mem, reward_mem, next_state_sequence_mem, next_label_sequence_mem))

    def act(self, state_sequence, label_sequence):
        # decision making (Random action)
        if np.random.rand() <= self.epsilon or len(state_sequence) < self.state_sequence_size:
            arr = np.random.rand(self.action_size)
            #print('Random action')
            return arr
        else:
            #print('Trained action')
            state_sequence = np.array(state_sequence)
            label_sequence = np.array(label_sequence)
            state_sequence = state_sequence.reshape((1, self.state_sequence_size, self.state_size,))
            label_sequence = label_sequence.reshape((1, self.label_sequence_size, self.num_label,))
            #print('[act]state_sequence'+str(state_sequence))
            #print('[act]label_sequence'+str(label_sequence))
            q_values = self.tar_model.predict([state_sequence, label_sequence])
            
            return q_values
    
    def act_trained(self, state_sequence, label_sequence):
        # decision making
        if len(state_sequence) < self.state_sequence_size:
            arr = np.random.rand(self.action_size)
            #print('Random action')
            return arr
        else:
            # Select the action for using the model
            state_sequence = np.array(state_sequence)
            label_sequence = np.array(label_sequence)
            state_sequence = state_sequence.reshape((1, self.state_sequence_size, self.state_size,))
            label_sequence = label_sequence.reshape((1, self.label_sequence_size, self.num_label,))
            
            q_values = self.eval_model.predict([state_sequence, label_sequence])
            
            return q_values
    
    def assign_memory(self):
        self.all_memory.append(self.memory)
        self.memory = deque(maxlen=int(1e6))
    
    def replay(self, num_episode, episode, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        train_X_states = []
        train_X_labels = []
        train_Y = []
        
        states_seq = []
        labels_seq = []
        actions = []
        rewards = []
        next_states_seq = []
        next_labels_seq = []
        
        for i in range(len(minibatch)):
            states_seq = minibatch[i][0]
            labels_seq = minibatch[i][1]
            actions = minibatch[i][2]
            rewards = minibatch[i][3]
            next_states_seq = minibatch[i][4]
            next_labels_seq = minibatch[i][5]
            
            arr_states_seq = np.array(states_seq)
            arr_labels_seq = np.array(labels_seq)
            arr_actions = np.array(actions)
            arr_rewards = np.array(rewards)
            arr_next_states_seq = np.array(next_states_seq)
            arr_next_labels_seq = np.array(next_labels_seq)
            
            # reshape for trainning input
            arr_states_seq = arr_states_seq.reshape((1, self.state_sequence_size, self.state_size,))
            arr_labels_seq = arr_labels_seq.reshape((1, self.label_sequence_size, self.num_label,))
            arr_next_states_seq = arr_next_states_seq.reshape((1, self.state_sequence_size, self.state_size,))
            arr_next_labels_seq = arr_next_labels_seq.reshape((1, self.label_sequence_size, self.num_label,))
            '''
            print('[replay]arr_states_seq: '+str(arr_states_seq))
            print('[replay]arr_labels_seq: '+str(arr_labels_seq))
            print('[replay]arr_actions: '+str(arr_actions))
            print('[replay]arr_rewards: '+str(arr_rewards))
            print('[replay]arr_next_states_seq: '+str(arr_next_states_seq))
            print('[replay]arr_next_labels_seq: '+str(arr_next_labels_seq))
            print('========================================================')
            '''
            #print('[replay]np.shape(arr_rewards): '+str(np.shape(arr_rewards)))
            #print('[replay]arr_rewards: '+str(arr_rewards))
            target = arr_rewards
            if arr_rewards!=10.:
                target = (arr_rewards + self.gamma * np.amax(self.tar_model.predict([arr_next_states_seq, arr_next_labels_seq])))
                #print('self.tar_model.predict(arr_next_states)): '+str(self.tar_model.predict([arr_next_states, arr_next_labels])))
                #print('np.amax(self.tar_model.predict(arr_next_states)): '+str(np.amax(self.tar_model.predict([arr_next_states, arr_next_labels]))))
            #print('[replay]target: '+str(target))
            target_f = self.eval_model.predict([arr_states_seq, arr_labels_seq])
            #print('[replay]original_target_f: '+str(target_f))

            #print('[replay]arr_actions: '+str(arr_actions))
            #print('[replay]np.shape(arr_actions): '+str(np.shape(arr_actions)))
            target_f[0][arr_actions] = target
            #print('[replay]updated_target_f: '+str(target_f))

            #self.model.fit([arr_states,arr_labels], target_f, epochs=1, verbose=0)

            train_X_states.append(arr_states_seq)
            train_X_labels.append(arr_labels_seq)
            train_Y.append(target_f[0])

            #print('[replay]arr_states'+str(arr_states))
            #print('[replay]arr_labels'+str(arr_labels))
            #print('[replay]target_f[0]'+str(target_f[0]))
        '''
        print('[replay]arr_train_X_states: '+str(arr_train_X_states))
        print('[replay]arr_train_X_labels: '+str(arr_train_X_labels))
        print('[replay]arr_train_Y: '+str(arr_train_Y))
        print('[replay]np.shape(train_X): '+str(np.shape(train_X)))
        print('[replay]np.shape(train_Y): '+str(np.shape(train_Y)))
        '''
        arr_train_X_states = np.array(train_X_states)
        arr_train_X_labels = np.array(train_X_labels)
        arr_train_Y = np.array(train_Y)
        #print('len(arr_train_X_states)'+str(len(arr_train_X_states)))
        #print('len(arr_train_Y)'+str(len(arr_train_Y)))
        arr_train_X_states = arr_train_X_states.reshape((len(arr_train_X_states), self.state_sequence_size, self.state_size,))
        arr_train_X_labels = arr_train_X_labels.reshape((len(arr_train_X_labels), self.label_sequence_size, self.num_label,))
        arr_train_Y = arr_train_Y.reshape((len(arr_train_Y), self.action_size))
        '''
        print('[replay_train]arr_train_X_states: '+str(arr_train_X_states))
        print('[replay_train]arr_train_X_labels: '+str(arr_train_X_labels))
        print('[replay_train]arr_train_Y: '+str(arr_train_Y))
        '''
        self.eval_model.fit([arr_train_X_states, arr_train_X_labels], arr_train_Y, epochs=1, verbose=0)
        
    def eval_to_tar(self):
        self.tar_model.set_weights(self.eval_model.get_weights())
        
    def decay(self, num_episode, episode):
        if self.epsilon > self.epsilon_min:
            #self.epsilon *= self.epsilon_decay
            self.epsilon = 1/math.exp(episode/(num_episode/12))
            #self.epsilon = 1/(1+math.exp(-(15/num_episode)*(num_episode/2-episode)))
        
    
    def load(self, name):
        self.eval_model.load_weights(name)

    def save(self, name):
        self.eval_model.save_weights(name)

        
