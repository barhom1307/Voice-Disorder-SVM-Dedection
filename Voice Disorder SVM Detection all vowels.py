# -*- coding: utf-8 -*-
"""
Created on Mar 23 2020

@author: Avinoam Barhom

Terms of use:
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND .
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY.
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
# from FeatureExtractionFunction import main_get_feature
import librosa
import pandas as pd
from scipy.stats import skew

# Dataset - http://stimmdb.coli.uni-saarland.de/index.php4#target
RATE = 16000 # 16KHz - same as in the article

# =============================================================================
#
# def get_files_per_sample(my_dir):
#     sampleDict = defaultdict(list)
#     for filename in os.listdir(my_dir):
#         sample_idx = filename.split("-")[0]
#         if 'iau' not in filename.split("-")[1]:
#             sampleDict[sample_idx].append(filename)
#     return sampleDict
#
# def get_mfcc_features(sampels, numCep, samplesNum):
#     mfccList = []
#     i = 0
#     while i < len(sampels):
#         tmp_arr = []
#         for arrVals in sampels[i:i+samplesNum]:
#             tmp = []
#             for value in arrVals:
#                 features = mfcc(np.frombuffer(value, dtype=np.int16), samplerate=RATE, numcep=numCep, winlen=0.008, winstep=0.008)
#                 print(features.shape)
#                 f_mean = np.mean(features, axis=0)
#                 f_std = np.std(features, axis=0)
#                 f_mean_std = np.dstack((f_mean,f_std)).ravel()
#                 assert f_mean_std.shape == (numCep*2,)
#                 tmp.append(f_mean_std)
#             tmp_arr.extend(tmp)
#         #  Normalize each key to a range of 0 to 1
#         tmp_arr = np.array(tmp_arr)
#         mfccList.append((tmp_arr-np.min(tmp_arr,axis=0))/(np.max(tmp_arr,axis=0)-np.min(tmp_arr,axis=0)))
#         i += samplesNum
#     return np.concatenate(mfccList, axis=0)
#
# def preprocess_files_per_speaker(my_dir, file_names):
#     signals = []
#     for file in file_names:
#         signal = AudioSegment.from_wav(my_dir + '\\' + file)
#         # Remove silence - beginning and end
#         non_sil_times = detect_nonsilent(signal, min_silence_len=50, silence_thresh=signal.dBFS * 1.5)
#         if len(non_sil_times): signal = signal[non_sil_times[0][0]:non_sil_times[0][1]]
#         # Downsampling to 16KHz
#         signal = signal.set_frame_rate(RATE)
#         # Wav segmentation
#         segmented_signal = wav_segmentation(signal)
#         segmented_signal = [chunk.get_array_of_samples() for chunk in segmented_signal]
#         signals.append(segmented_signal)
#     return signals
#
# def iter_speakers(my_dir, sampleDict, n_iters=2):
#     sampels = []
#     for speaker in sampleDict.keys():
#         signals = preprocess_files_per_speaker(my_dir, sampleDict[speaker])
#         for n in range(n_iters):
#             tmp = []
#             for vowel_list in signals:
#                 rand_idx = np.random.randint(len(vowel_list))
#                 tmp.append(vowel_list[rand_idx])
#             sampels.append(tmp)
#     return sampels
#
# def preprocess_samples(my_dir, dictsList, numCep, iter_num=5):
#     features_per_pitch_dict = []
#     for dictL in dictsList:
#         features_per_speaker = []
#         for speakerNum in dictL.keys():
#             file_features = []
#             for file in dictL[speakerNum]:
#                 signal = AudioSegment.from_wav(my_dir + '\\' + file)
#                 # Remove silence - beginning and end
#                 non_sil_times = detect_nonsilent(signal, min_silence_len=50, silence_thresh=signal.dBFS * 1.5)
#                 if len(non_sil_times): signal = signal[non_sil_times[0][0]:non_sil_times[0][1]]
#                 # Downsampling to 16KHz
#                 signal = signal.set_frame_rate(RATE)
#                 # Wav segmentation
#                 segmented_signal = wav_segmentation(signal)
#                 segmented_signal = [chunk.get_array_of_samples() for chunk in segmented_signal]
#                 tmp = []
#                 for arr in segmented_signal:
#                     features = mfcc(np.frombuffer(arr, dtype=np.int16), samplerate=RATE, numcep=numCep, winlen=0.008, winstep=0.008)
#                     f_mean = np.mean(features, axis=0)
#                     f_std = np.std(features, axis=0)
#                     f_mean_std = np.dstack((f_mean,f_std)).ravel()
#                     tmp.append(f_mean_std)
#                 file_features.append(tmp)
#             iter_features_per_speaker = []
#             for n in range(iter_num):
#                 tmp = []
#                 for featuresList in file_features:
#                     rand_idx = np.random.randint(len(featuresList))
#                     tmp.append(featuresList[rand_idx])
#                 iter_features_per_speaker.append(np.concatenate(tmp))
#             features_per_speaker.append(np.array(iter_features_per_speaker))
#         features_per_pitch_dict.append(np.concatenate(features_per_speaker))
#     tmp = np.concatenate(features_per_pitch_dict)
#     data_normalized = (tmp-np.min(tmp,axis=0))/(np.max(tmp,axis=0)-np.min(tmp,axis=0))
#     return data_normalized
#
# =============================================================================

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

def get_files_per_sample_pitch(my_dir):
    sampleLow, sampleNormal, sampleHigh = defaultdict(list), defaultdict(list), defaultdict(list)
    dicts = {'h': sampleHigh, 'n': sampleNormal, 'l': sampleLow}
    for filename in os.listdir(my_dir):
        sampleNum = filename.split("-")[0]
        if '_' in filename:
            pitch = filename.split("_")[1][0]
            dicts[pitch][sampleNum].append(filename)
    return sampleLow, sampleNormal, sampleHigh

def get_segmented_samples(my_dir, dictsList, iter_num=5):
    all_pitchs = []
    for dictL in dictsList:
        all_speakers = []
        for speakerNum in dictL.keys():
            segments = []
            for file in dictL[speakerNum]:
                signal = AudioSegment.from_wav(my_dir + '\\' + file)
                # Remove silence - beginning and end
                non_sil_times = detect_nonsilent(signal, min_silence_len=50, silence_thresh=signal.dBFS * 1.5)
                if len(non_sil_times): signal = signal[non_sil_times[0][0]:non_sil_times[0][1]]
                # Downsampling to 16KHz
                signal = signal.set_frame_rate(RATE)
                # Wav segmentation
                segmented_signal = wav_segmentation(signal)
                segmented_signal = [chunk.get_array_of_samples() for chunk in segmented_signal]
                segments.append(segmented_signal)
            iter_segments_per_speaker = []
            for n in range(iter_num):
                tmp = []
                for segList in segments:
                    rand_idx = np.random.randint(len(segList))
                    tmp.append(segList[rand_idx])
                iter_segments_per_speaker.append(tmp)
            all_speakers.append(iter_segments_per_speaker)
        all_pitchs.append(all_speakers)
    return all_pitchs

def get_mfcc_features_all_vowels(all_pitchs, numCep):
    all_features = []
    for pitchList in all_pitchs:
        speaker_features = []
        for speakerList in pitchList:
            file_features = []
            for segList in speakerList:
                tmp = []
                for arr in segList:
                    features = mfcc(np.frombuffer(arr, dtype=np.int16), samplerate=RATE, numcep=numCep, winlen=0.008, winstep=0.008)
                    f_mean = np.mean(features, axis=0)
                    # f_std = np.std(features, axis=0)
                    # f_mean_std = np.dstack((f_mean,f_std)).ravel()
                    tmp.append(f_mean)
                file_features.append(np.concatenate(tmp))
            speaker_features.append(np.array(file_features))
        all_features.append(np.concatenate(speaker_features))
    tmp = np.concatenate(all_features)
    data_normalized = (tmp-np.min(tmp,axis=0))/(np.max(tmp,axis=0)-np.min(tmp,axis=0))
    return data_normalized

def get_vowels_dataset(neg_train, neg_test, pos_train, pos_test):
    # test_ratio = 20%
    print("Building dataset...")
    # Train data
    v_neg = np.zeros((len(neg_train),1))
    v_pos = np.ones((len(pos_train),1))
    v_neg = np.concatenate((neg_train, v_neg), axis=1)
    v_pos = np.concatenate((pos_train, v_pos), axis=1)
    train_data = np.concatenate((v_neg, v_pos))
    np.random.shuffle(train_data)
    # Test data
    v_neg = np.zeros((len(neg_test),1))
    v_pos = np.ones((len(pos_test),1))
    v_neg = np.concatenate((neg_test, v_neg), axis=1)
    v_pos = np.concatenate((pos_test, v_pos), axis=1)
    test_data = np.concatenate((v_neg, v_pos))
    np.random.shuffle(test_data)
    return train_data, test_data

def plt_SVM(numCep, accList, farList, k):
    fig, ax1 = plt.subplots()
    plt.title(f"Voice Disorder SVM Detection, kernel={k}\nAccuracy as function of Mel Frequency Cepstral Coefficient (MFCC) amount\nUsing Saarbruecken Voice Database", pad=10, weight='bold')
    ax1.plot(numCep, accList, '-r', label='Vowels \'aiu\' Accuracy')
    ax1.set_xlabel("Number of Coefficients")
    ax1.set_ylabel("Balanced Accuracy Score")
    # ax1.set_ylim(np.min(accList)-0.005, 1.005)
    ax1.set_ylim(0, 1.005)
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(numCep, farList, ':r', label='Vowels \'aiu\' Far')
    ax2.set_ylabel("False Alarms")
    # ax2.set_ylim(np.min(farList)-0.005, np.max(farList)+0.005)
    ax2.set_ylim(0, 1.005)
    fig.tight_layout()
    plt.legend(loc='upper right')
    plt.show()

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

def SVM_test(train_data, test_data, k='linear'):
    print("Train and test linear SVM on each vowel...")
    print('#'*10 + ' '*6 + "Vowels \'aiu\' results" + ' '*6 + '#'*10)
    svclassifier = SVC(kernel=k, class_weight='balanced')
    svclassifier.fit(train_data[:,:-1], train_data[:,-1])
    with open('SVM_all.pkl', 'wb') as fid:
        pkl.dump(svclassifier, fid)
    y_pred = svclassifier.predict(test_data[:,:-1])
    acc = balanced_accuracy_score(test_data[:,-1], y_pred)
    far = np.sum(y_pred[np.argwhere(test_data[:,-1]==0)])/len(np.argwhere(test_data[:,-1]==0))
    print("Confidence test for vowels \'iau\':")
    print(calc_confidence_interval(test_data[:,-1], y_pred))
    return acc, far

def read_samples(my_dir, iterNum):
    print("Preprocessing negative samples...")
    sampleLow, sampleNormal, sampleHigh = get_files_per_sample_pitch(my_dir + '\\train_neg')
    pitchs_train_neg = get_segmented_samples(my_dir + '\\train_neg', [sampleLow, sampleNormal, sampleHigh], iterNum)
    sampleLow, sampleNormal, sampleHigh = get_files_per_sample_pitch(my_dir + '\\test_neg')
    pitchs_test_neg = get_segmented_samples(my_dir + '\\test_neg', [sampleLow, sampleNormal, sampleHigh], iterNum)
    print("Preprocessing positive samples...")
    sampleLow, sampleNormal, sampleHigh = get_files_per_sample_pitch(my_dir + '\\train_pos')
    pitchs_train_pos = get_segmented_samples(my_dir + '\\train_pos', [sampleLow, sampleNormal, sampleHigh], iterNum)
    sampleLow, sampleNormal, sampleHigh = get_files_per_sample_pitch(my_dir + '\\test_pos')
    pitchs_test_pos = get_segmented_samples(my_dir + '\\test_pos', [sampleLow, sampleNormal, sampleHigh], iterNum)
    return pitchs_train_neg, pitchs_test_neg, pitchs_train_pos, pitchs_test_pos


requested_speaker_iters = 10 # requested loops per total segmented samples of spcific speaker
numCep = np.arange(10,20,1) # Mel Frequency Cepstral Coefficient amount
# numCep = np.array([20])
my_dir = r"C:\Users\R\Desktop\Voice_Disorder_SVM_Dedection\dataset"
kernel = 'linear'
accList, farList = [], []
max_acc, far_max, nCep_max = 0, 0, 0
pitchs_train_neg, pitchs_test_neg, pitchs_train_pos, pitchs_test_pos = read_samples(my_dir, requested_speaker_iters)
for n in numCep:
    print(f"Eval numcep = {n}")
    neg_train = get_mfcc_features_all_vowels(pitchs_train_neg, n)
    neg_test = get_mfcc_features_all_vowels(pitchs_test_neg, n)
    pos_train = get_mfcc_features_all_vowels(pitchs_train_pos, n)
    pos_test = get_mfcc_features_all_vowels(pitchs_test_pos, n)
    train_data, test_data = get_vowels_dataset(neg_train, neg_test, pos_train, pos_test)
    acc, far = SVM_test(train_data, test_data, kernel)
    if acc > max_acc:
        max_acc = acc
        nCep_max = n
        far_max = far
    accList.append(acc)
    farList.append(far)
plt_SVM(numCep, accList, farList, kernel)
print("SVM detection using MFCC feature only")
print(f"Max Balanced Accuracy Score of Vowels \'aiu\' is - {np.round((max_acc)*100,2)}% , with Far = {np.round((far_max)*100,2)}% and {int(nCep_max)} Coefficients" )

