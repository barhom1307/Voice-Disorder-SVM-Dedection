# -*- coding: utf-8 -*-
"""
Created on Mar 23 2020

@author: Avinoam Barhom

Terms of use:
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND .
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from pydub import AudioSegment
from collections import defaultdict
import os
from pydub.silence import detect_nonsilent
from sklearn.utils import class_weight


RATE = 16000 # 16KHz - same as in the article

def build_model(features_size):
    input_features = Input(shape=(features_size,), name="input_features")
    dense_1 = Dense(features_size//100, activation='relu', name='dense1')(input_features)
    # dense_2 = Dense(features_size//1000, activation='relu', name='dense2')(dense_1)
    output = Dense(1, activation='softmax', name='output')(dense_1)
    model = Model(input_features, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def read_samples(my_dir):
    print("Preprocessing negative samples...")
    neg_train = preprocess_wav_files(my_dir + '/train_neg')
    neg_test = preprocess_wav_files(my_dir + '/test_neg')
    print("Preprocessing positive samples...")
    pos_train = preprocess_wav_files(my_dir + '/train_pos')
    pos_test = preprocess_wav_files(my_dir + '/test_pos')
    del neg_train['iau'] # deleting unused key - samples of all vowels together ('iau')
    del neg_test['iau'] # deleting unused key - samples of all vowels together ('iau')
    return neg_train, neg_test, pos_train, pos_test

def preprocess_wav_files(my_dir):
    vowelsDict = defaultdict(list)
    for filename in os.listdir(my_dir):
        vowel = filename.split("_")[0][-1]
        # File read
        signal = AudioSegment.from_wav(my_dir + '/' + filename)
        # Remove silence - beginning and end
        non_sil_times = detect_nonsilent(signal, min_silence_len=50, silence_thresh=signal.dBFS * 1.5)
        if len(non_sil_times): signal = signal[non_sil_times[0][0]:non_sil_times[0][1]]
        # Downsampling to 16KHz
        signal = signal.set_frame_rate(RATE)
        # Wav segmentation
        segmented_signal = wav_segmentation(signal)
        tmp_seg = np.array([np.array(chunk.get_array_of_samples()) for chunk in segmented_signal])
        # tmp_seg = np.array([np.frombuffer(value, dtype=np.float16) for value in segmented_signal])
        if '_' in filename:
            vowelsDict[vowel].extend(tmp_seg)
            # librosa_data[vowel].append(librosa_features(my_dir + '/' + filename))
        else:
            vowelsDict['iau'].extend(segmented_signal)
    return vowelsDict

def wav_segmentation(signal, frame_len=500, overlap=400):
# =============================================================================
#     Each file is segmented into multiple 500 ms long snippets,
#     with a 400 ms overlap of subsequent snippets.
#     Please note - there is padding silence at the last frame.
# =============================================================================
    diff = frame_len-overlap
    # check if the signal is shorter then requested frame length
    if len(signal) <= frame_len:
            silence = AudioSegment.silent(duration=frame_len-len(signal))
            output = [signal + silence]
            return output
    loop_cnt = 1+ ((len(signal)-frame_len)//diff)
    output = []
    for i in range(loop_cnt):
        output.append(signal[i*diff:frame_len+i*diff])
    silence = AudioSegment.silent(duration=diff-len(signal)%diff)
    # check if there isn't too much silence (50% of last overlap)
    if len(silence) < diff//2:
        output.append(signal[(i+1)*diff:]+silence)
    return output

def get_vowels_dataset(dataset):
    amount = 7995
    for tList in dataset:
        for key in tList.keys():
            tList[key] = np.array([arr[:amount] for arr in tList[key]])
    # test_ratio = 20%
    neg_train, neg_test, pos_train, pos_test = dataset
    vowels_dataset = defaultdict(list)
    print("Building dataset for each vowel...")
    for key in neg_train.keys():
        # Train data
        v_neg = np.zeros((len(neg_train[key]),1))
        v_pos = np.ones((len(pos_train[key]),1))
        v_neg = np.concatenate((neg_train[key], v_neg), axis=1)
        v_pos = np.concatenate((pos_train[key], v_pos), axis=1)
        train_data = np.concatenate((v_neg, v_pos))
        np.random.shuffle(train_data)
        # Test data
        v_neg = np.zeros((len(neg_test[key]),1))
        v_pos = np.ones((len(pos_test[key]),1))
        v_neg = np.concatenate((neg_test[key], v_neg), axis=1)
        v_pos = np.concatenate((pos_test[key], v_pos), axis=1)
        test_data = np.concatenate((v_neg, v_pos))
        np.random.shuffle(test_data)
        # Normalize
        tmp_train, tmp_test = np.array(train_data[:,:-1]), np.array(test_data[:,:-1])
        pred_train, pred_test = np.reshape(train_data[:,-1], (len(train_data[:,-1]),1)), np.reshape(test_data[:,-1], (len(test_data[:,-1]),1))
        tmp_train = (tmp_train-np.min(tmp_train,axis=0))/(np.max(tmp_train,axis=0)-np.min(tmp_train,axis=0))
        tmp_test = (tmp_test-np.min(tmp_test,axis=0))/(np.max(tmp_test,axis=0)-np.min(tmp_test,axis=0))
        train_data = np.concatenate((tmp_train, pred_train), axis=1)
        test_data = np.concatenate((tmp_test, pred_test), axis=1)
        vowels_dataset[key] = [train_data, test_data]
    return vowels_dataset

if __name__== '__main__':
    my_dir = r"C:\Users\R\Desktop\Voice_Disorder_SVM_Dedection\dataset"
    if os.path.isfile(my_dir + '\\dataset.npy'):
        dataset = np.load(my_dir + '\\dataset.npy', allow_pickle=True)
    else:
        neg_train, neg_test, pos_train, pos_test = read_samples(my_dir)
        dataset = [neg_train, neg_test, pos_train, pos_test]
        np.save(my_dir + '\\dataset.npy', dataset)
    vowels_dataset = get_vowels_dataset(dataset)
    features_size = vowels_dataset['a'][0].shape[1] - 1
    vowel_models = []
    preds = []
    for key in vowels_dataset.keys():
        train_data, test_data = vowels_dataset[key]
        class_weights = class_weight.compute_class_weight('balanced', np.unique(train_data[:,-1]), train_data[:,-1])
        dense_model = build_model(features_size)
        dense_model.fit(train_data[:,:-1], train_data[:,-1], epochs=100, batch_size=32, validation_split=0.2, class_weight=class_weights)
        y_pred = dense_model.predict(test_data[:,:-1])
        vowel_models.append(dense_model)
        preds.append(y_pred)


