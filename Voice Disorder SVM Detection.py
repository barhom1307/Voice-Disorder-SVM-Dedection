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
from sklearn.metrics import classification_report, accuracy_score


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
    negative_samples = preprocess_wav_files(my_dir + '/negative_samples')
    print("Preprocessing positive samples...")
    positive_samples = preprocess_wav_files(my_dir + '/positive_samples')
    del negative_samples['iau'] # deleting unused key - samples of all vowels together ('iau')
    return negative_samples, positive_samples
    
def get_vowels_dataset(negative_samples, positive_samples, numcep=20, test_ratio=0.2):
    print("Extracting negative mfcc features...")
    mfcc_negative = get_mfcc_features(negative_samples, numcep)
    print("Extracting positive mfcc features...")
    mfcc_positive = get_mfcc_features(positive_samples, numcep)
    vowels_dataset = defaultdict(list)
    print("Building dataset for each vowel...")
    for key in mfcc_negative.keys():
        v_neg = np.zeros((len(mfcc_negative[key]),1))
        v_pos = np.ones((len(mfcc_positive[key]),1))
        v_neg = np.concatenate((mfcc_negative[key], v_neg), axis=1)
        v_pos = np.concatenate((mfcc_positive[key], v_pos), axis=1)
        dataset = np.concatenate((v_neg, v_pos))
        np.random.shuffle(dataset)
        train_data = dataset[:int(len(dataset)*(1-test_ratio))] 
        test_data = dataset[int(len(dataset)*(1-test_ratio)):] 
        vowels_dataset[key] = [train_data, test_data]
    return vowels_dataset

def SVM_test(dataset):
    print("Train and test linear SVM on each vowel...")
    accList = []
    for key in dataset.keys():
        # print('#'*49)
        print('#'*10 + ' '*6 + f"Vowel \'{key}\' results" + ' '*6 + '#'*10)
        # print('#'*49)
        train, test = vowels_dataset[key]
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(train[:,:-1], train[:,-1])
        y_pred = svclassifier.predict(test[:,:-1])
        # print(classification_report(test[:,-1], y_pred))
        acc = accuracy_score(test[:,-1], y_pred)
        accList.append(acc)
    return accList

def plt_SVM(numCep, accList):
    plt.figure()
    plt.plot(numCep, accList[:,0], '-r', label='Vowel \'a\'')
    plt.plot(numCep, accList[:,1], '-g', label='Vowel \'i\'')
    plt.plot(numCep, accList[:,2], '-b', label='Vowel \'u\'')
    plt.title("Voice Disorder SVM Detection\nAccuracy as function of Mel Frequency Cepstral Coefficient (MFCC) amount\nUsing Saarbruecken Voice Database", pad=10, weight='bold')
    plt.xlabel("Number of Coefficients")
    plt.ylabel("Accuracy Score")
    plt.ylim(0.98, 1.005)
    plt.legend()
    plt.show()
    
def test_person(my_dir, person_dir, vowels_dataset):
    vowelsDict = preprocess_wav_files(my_dir + person_dir)
    mfcc = get_mfcc_features(vowelsDict, 20)
    for key in vowels_dataset.keys():
        train, test = vowels_dataset[key]
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(train[:,:-1], train[:,-1])
        y_pred = svclassifier.predict(mfcc[key])
        acc = accuracy_score(np.zeros((len(y_pred),1)), y_pred)
        print(f"According to Vowel \'{key}\' you are {np.round((1-acc)*100,2)} % sick")

if __name__== '__main__':
    my_dir = 'C:/Users/Avinoam/Desktop'
    negative_samples, positive_samples = read_samples(my_dir)
    numCep = np.arange(10,21,1) # Mel Frequency Cepstral Coefficient amount
    accList = []
    for n in numCep:  
        print(f"Eval numcep = {n}")
        vowels_dataset = get_vowels_dataset(negative_samples, positive_samples, numcep=n)
        # np.save(my_dir + '/dataset.npy', vowels_dataset)
        acc = SVM_test(vowels_dataset)
        accList.append(acc)
    res = np.reshape(accList, (len(numCep),3))
    plt_SVM(numCep, res)
    test_person(my_dir,'/oshri_voice',vowels_dataset)
