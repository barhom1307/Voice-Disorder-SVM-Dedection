# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:36:19 2020

@author: Avinoam
"""

import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import librosa.display as rosaplt
from pydub.silence import detect_nonsilent
from pydub import AudioSegment
from python_speech_features import mfcc
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
import pickle as pkl
from FeatureExtractionFunction import main_get_feature
import librosa
import pandas as pd
from scipy.stats import skew

# Dataset - http://stimmdb.coli.uni-saarland.de/index.php4#target 
RATE = 16000 # 16KHz - same as in the article 

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
    
def librosa_features(file):
    data, _ = librosa.core.load(file, sr = 44100)
    ft1 = librosa.feature.zero_crossing_rate(data)[0]
    ft2 = librosa.feature.spectral_rolloff(data)[0]
    ft3 = librosa.feature.spectral_centroid(data)[0]
    ft4 = librosa.feature.spectral_contrast(data)[0]
    ft5 = librosa.feature.spectral_bandwidth(data)[0]
    ft1_trunc = np.hstack((np.mean(ft1), np.std(ft1), skew(ft1), np.max(ft1), np.median(ft1), np.min(ft1)))
    ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
    ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
    ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
    ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
    return np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc))
    
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
        segmented_signal = [chunk.get_array_of_samples() for chunk in segmented_signal]
        if '_' in filename:
            vowelsDict[vowel].extend(segmented_signal)
            # librosa_data[vowel].append(librosa_features(my_dir + '/' + filename))
        else:
            vowelsDict['iau'].extend(segmented_signal)
    return vowelsDict

def plt_mfcc(mfcc_features):
    plt.figure(figsize=(10, 4))
    rosaplt.specshow(mfcc_features, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

def get_mfcc_features(samples_dict, numCep):
    mfccDict = defaultdict(list)
    for key in samples_dict.keys():
        for value in samples_dict[key]:
            # numcep=20, winlen=0.008, winstep=0.008 - in order to get 20*63 input feature matrix (as mentioned in the article)
            features = mfcc(np.frombuffer(value, dtype=np.int16), samplerate=RATE, numcep=numCep, winlen=0.008, winstep=0.008)
            # assert features.shape == (63,numCep)
            f_mean = np.mean(features, axis=0)
            f_std = np.std(features, axis=0)
            f_mean_std = np.dstack((f_mean,f_std)).ravel()
            assert f_mean_std.shape == (numCep*2,)
            mfccDict[key].append(f_mean_std)
        #  Normalize each key to a range of 0 to 1
        tmp = np.array(mfccDict[key])
        mfccDict[key] = (tmp-np.min(tmp,axis=0))/(np.max(tmp,axis=0)-np.min(tmp,axis=0))
        # mfccDict[key] = preprocessing.normalize(mfccDict[key], axis=0)
    return mfccDict      

def read_samples(my_dir):
    print("Preprocessing negative samples...")
    neg_train = preprocess_wav_files(my_dir + '/train_neg')
    neg_test = preprocess_wav_files(my_dir + '/test_neg')
    print("Preprocessing positive samples...")
    pos_train = preprocess_wav_files(my_dir + '/train_pos')
    pos_test = preprocess_wav_files(my_dir + '/test_pos')
    # del neg_train['iau'] # deleting unused key - samples of all vowels together ('iau')
    # del neg_test['iau'] # deleting unused key - samples of all vowels together ('iau')
    return neg_train, neg_test, pos_train, pos_test
    
def get_vowels_dataset(neg_train, neg_test, pos_train, pos_test, numcep=20):
    # test_ratio = 20%
    print("Extracting negative mfcc features...")
    mfcc_negative_train = get_mfcc_features(neg_train, numcep)
    mfcc_negative_test = get_mfcc_features(neg_test, numcep)
    print("Extracting positive mfcc features...")
    mfcc_positive_train = get_mfcc_features(pos_train, numcep)
    mfcc_positive_test = get_mfcc_features(pos_test, numcep)
    vowels_dataset = defaultdict(list)
    print("Building dataset for each vowel...")
    for key in mfcc_negative_train.keys():
        # Train data
        v_neg = np.zeros((len(mfcc_negative_train[key]),1))
        v_pos = np.ones((len(mfcc_positive_train[key]),1))
        v_neg = np.concatenate((mfcc_negative_train[key], v_neg), axis=1)
        v_pos = np.concatenate((mfcc_positive_train[key], v_pos), axis=1)
        train_data = np.concatenate((v_neg, v_pos))
        np.random.shuffle(train_data)
        # Test data
        v_neg = np.zeros((len(mfcc_negative_test[key]),1))
        v_pos = np.ones((len(mfcc_positive_test[key]),1))
        v_neg = np.concatenate((mfcc_negative_test[key], v_neg), axis=1)
        v_pos = np.concatenate((mfcc_positive_test[key], v_pos), axis=1)
        test_data = np.concatenate((v_neg, v_pos))
        np.random.shuffle(test_data)
        vowels_dataset[key] = [train_data, test_data]
    return vowels_dataset

def SVM_test(dataset, k='linear'):
    print("Train and test linear SVM on each vowel...")
    accList, farList = [], []
    all_train, all_test = [], []
    for key in dataset.keys():
        print('#'*10 + ' '*6 + f"Vowel \'{key}\' results" + ' '*6 + '#'*10)
        train, test = dataset[key]
        all_train.append(train)
        all_test.append(test)
        svclassifier = SVC(kernel=k, class_weight='balanced')
        # svclassifier = SVC(kernel=k)
        svclassifier.fit(train[:,:-1], train[:,-1])
        with open(f'SVM_{key}.pkl', 'wb') as fid:
            pkl.dump(svclassifier, fid)
        y_pred = svclassifier.predict(test[:,:-1])
        # print(classification_report(test[:,-1], y_pred))
        acc = balanced_accuracy_score(test[:,-1], y_pred)
        far = np.sum(y_pred[np.argwhere(test[:,-1]==0)])/len(np.argwhere(test[:,-1]==0))
        accList.append(acc)
        farList.append(far)
        print(f"Confidence test for vowel \'{key}\'':")
        print(calc_confidence_interval(test[:,-1], y_pred))
    all_train = np.concatenate(all_train)
    all_test = np.concatenate(all_test)
    np.random.shuffle(all_train)
    np.random.shuffle(all_test)
    print('#'*10 + ' '*6 + "Vowels \'aiu\' results" + ' '*6 + '#'*10)
    svclassifier = SVC(kernel=k, class_weight='balanced')
    # svclassifier = SVC(kernel=k)
    svclassifier.fit(all_train[:,:-1], all_train[:,-1])
    with open('SVM_all.pkl', 'wb') as fid:
        pkl.dump(svclassifier, fid)
    y_pred = svclassifier.predict(all_test[:,:-1])
    acc = balanced_accuracy_score(all_test[:,-1], y_pred)
    far = np.sum(y_pred[np.argwhere(all_test[:,-1]==0)])/len(np.argwhere(all_test[:,-1]==0))
    accList.append(acc)
    farList.append(far)
    print("Confidence test for vowels \'iau\':")
    print(calc_confidence_interval(all_test[:,-1], y_pred))
    return accList, farList
    
def plt_SVM(numCep, accList, farList, k):
    fig, ax1 = plt.subplots()
    plt.title(f"Voice Disorder SVM Detection, kernel={k}\nAccuracy as function of Mel Frequency Cepstral Coefficient (MFCC) amount\nUsing Saarbruecken Voice Database", pad=10, weight='bold')
    ax1.plot(numCep, accList[:,0], '-r', label='Vowel \'a\' Accuracy')
    ax1.plot(numCep, accList[:,1], '-g', label='Vowel \'i\' Accuracy')
    ax1.plot(numCep, accList[:,2], '-b', label='Vowel \'u\' Accuracy')
    ax1.plot(numCep, accList[:,3], '-k', label='Vowels \'aiu\' Accuracy')
    ax1.set_xlabel("Number of Coefficients")
    ax1.set_ylabel("Balanced Accuracy Score")    
    # ax1.set_ylim(np.min(accList)-0.005, 1.005)
    ax1.set_ylim(0, 1.005)
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(numCep, farList[:,0], ':r', label='Vowel \'a\' Far')
    ax2.plot(numCep, farList[:,1], ':g', label='Vowel \'i\' Far')
    ax2.plot(numCep, farList[:,2], ':b', label='Vowel \'u\' Far')
    ax2.plot(numCep, farList[:,3], ':k', label='Vowels \'aiu\' Far')    
    ax2.set_ylabel("False Alarms") 
    # ax2.set_ylim(np.min(farList)-0.005, np.max(farList)+0.005)
    ax2.set_ylim(0, 1.005)
    fig.tight_layout()
    plt.legend(loc='upper right')
    plt.show()

def test_person(my_dir, person_dir, vowels_dataset):
    vowelsDict = preprocess_wav_files(my_dir + person_dir)
    mfcc = get_mfcc_features(vowelsDict, 20)
    for key in vowels_dataset.keys():
        if key in mfcc.keys():
            with open(f'SVM_{key}.pkl', 'rb') as f:
                svclassifier = pkl.load(f)
            y_pred = svclassifier.predict(mfcc[key])
            res = len(np.argwhere(y_pred==1))/len(y_pred)
            print(f"According to Vowel \'{key}\' you are {np.round(res*100,2)}% sick")

def test_scalar_features(my_dir):
    # Please note that dict from "main_get_feature" method is normalized
    if os.path.exists(my_dir + '/scalar_features_dataset.npy'):
        print("Loading dataset...")
        neg_train, neg_test, pos_train, pos_test = np.load(my_dir + '/scalar_features_dataset.npy', allow_pickle=True)
    else:
        print("Get negative samples features...")
        neg_train = main_get_feature(my_dir + '/train_neg')
        neg_test = main_get_feature(my_dir + '/test_neg')
        print("Get positive samples features...")
        pos_train = main_get_feature(my_dir + '/train_pos')
        pos_test = main_get_feature(my_dir + '/test_pos')
        np.save(my_dir + '/scalar_features_dataset.npy', [neg_train, neg_test, pos_train, pos_test])
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
        vowels_dataset[key] = [train_data, test_data]
    return vowels_dataset
    
def calc_confidence_interval(y_true, y_pred):
    const_prob = {'90%':1.64, '95%':1.96, '98%':2.33, '99%':2.58}
    sentence = ''
    for key in const_prob.keys():
        # classification error = incorrect predictions / total predictions
        error = sum(y_true != y_pred)/len(y_pred)
        prob = const_prob[key] * np.sqrt((error*(1-error))/len(y_pred))
        lim = np.clip(np.array([np.round(error - prob,3), np.round(error + prob,3)]), 0.0, 1.0)
        sentence += f"With {key} Confidence, the confidence interval is [{lim[0]},{lim[1]}]\n"
    return sentence

def test_SVM_with_mfcc(my_dir, kernel):
    max_acc = np.zeros(4)
    nCep_max = np.zeros(4)
    far_max = np.zeros(4)
    if os.path.exists(my_dir + '/features_dataset.npy'):
        print("Loading dataset...")
        neg_train, neg_test, pos_train, pos_test = np.load(my_dir + '/features_dataset.npy', allow_pickle=True)
    else:
        neg_train, neg_test, pos_train, pos_test = read_samples(my_dir)
        np.save(my_dir + '/features_dataset.npy', [neg_train, neg_test, pos_train, pos_test])
    numCep = np.arange(10,21,1) # Mel Frequency Cepstral Coefficient amount
    # numCep = np.array([20])
    accList, farList = [], []
    for n in numCep:  
        print(f"Eval numcep = {n}")
        vowels_dataset = get_vowels_dataset(neg_train, neg_test, pos_train, pos_test, numcep=n)
        acc, far = SVM_test(vowels_dataset, kernel)
        for i in range(len(max_acc)):
            if acc[i] > max_acc[i]: 
                max_acc[i] = acc[i]
                nCep_max[i] = n
                far_max[i] = far[i]
        accList.append(acc)
        farList.append(far)
    res_acc = np.reshape(accList, (len(accList),len(accList[0])))
    res_far = np.reshape(farList, (len(farList),len(farList[0])))
    plt_SVM(numCep, res_acc, res_far, kernel)
    print("SVM detection using MFCC feature only")
    for i,key in enumerate(vowels_dataset.keys()):
        print(f"Max Balanced Accuracy Score of Vowel \'{key}\' is - {np.round((max_acc[i])*100,2)}% , with Far = {np.round((far_max[i])*100,2)}% and {int(nCep_max[i])} Coefficients" ) 
    print(f"Max Balanced Accuracy Score of Vowels \'aiu\' is - {np.round((max_acc[i+1])*100,2)}% , with Far = {np.round((far_max[i+1])*100,2)}% and {int(nCep_max[i+1])} Coefficients" ) 
    return vowels_dataset


if __name__== '__main__':
    my_dir = 'C:/Users/Avinoam/Desktop/Voice-Disorder-SVM-Dedection/dataset'
    featuers = 'mfcc' # 'all'- takes all scalars global featurs, 'mfcc' - takes mfcc feature alone
    kernel = 'linear'
    if featuers == 'mfcc':
        vowels_dataset = test_SVM_with_mfcc(my_dir, kernel)
        # test_person('C:/Users/Avinoam/Desktop','/oshri_voice',vowels_dataset)
    elif featuers == 'all':
        vowels_dataset = test_scalar_features(my_dir)
        acc, far = SVM_test(vowels_dataset)
        print("SVM detection using all global scalars features")
        for i,key in enumerate(vowels_dataset.keys()):
            print(f"Accuracy Score of Vowel \'{key}\' is - {np.round(acc[i]*100,2)}% with FAR={np.round(far[i]*100,2)}%")