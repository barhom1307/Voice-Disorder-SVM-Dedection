# -*- coding: utf-8 -*-
"""
Created on Mar 23 2020

@author: Daniel Teitelman

Terms of use:
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND .
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY.
"""

# %% Requirements

'''
In order to run FeatureExtractionFunction.py follow the follwing instructions:
    1. In console: pip install librosa.
    2. Download Praat from the follwing link: http://www.fon.hum.uva.nl/praat/download_win.html no need to install Phonetic and international symbols or anything else.
    3. Unzip the zip you downloaded and place praat.exe in your desktop.
    4. In console: pip install praat-parselmouth.
    5. In console: pip install nolds. !!optional!! for chaos theory based features dont install it at first. In addition you might need an additional library named quantumrandom.
    6. Profit.
'''

# %% Imports

import numpy as np
import scipy
import librosa
import parselmouth
from scipy.io import wavfile
from parselmouth.praat import call
from scipy import signal
import nolds
from scipy.signal import hilbert
import os
from collections import defaultdict


# %% Global Variables

scalar = 'scalar'
vector = 'vector'
matrix = 'matrix'

# %% Functions

def get_features_of_type(features, features_type,type_feature):
    ''' 
    Returns a list of features of certain type
    '''
    features_of_type = []
    for i in range(len(features_type)):
        if features_type[i] == type_feature:
            features_of_type.append(features[i])
    return features_of_type
        

def get_instantaneous_freq(signal):
    ''' 
    Calculates instantaneous frequency by hilbert transfrom
    implemented from:
    https://dsp.stackexchange.com/questions/25845/meaning-of-hilbert-transform
    '''
    hilbert_trans = hilbert(signal)
    ins_freq = (np.gradient(np.angle(hilbert_trans))) / (np.pi * 2)
    return ins_freq
    

def get_entropy(signal): 
    ''' 
    Calculates two channel entropy by histogram method
    '''
    hist1 = np.histogram(signal[:,0],np.max(signal))
    hist2 = np.histogram(signal[:,1],np.max(signal))
    hist1_dist = scipy.stats.rv_histogram(hist1).pdf(np.linspace(0,np.max(signal),np.max(signal)+1))
    hist2_dist = scipy.stats.rv_histogram(hist2).pdf(np.linspace(0,np.max(signal),np.max(signal)+1))
    entropyLeftChannel = scipy.stats.entropy(hist1_dist)
    entropyRightChannel = scipy.stats.entropy(hist2_dist)
    return entropyLeftChannel, entropyRightChannel

# sound  = .wav sound file
# sr = sample rate usually for wav its 44.1k , sr is extracted by scipy and used by librosa

def get_features(sound_lib,sr,sound_scipy,sound_praat):
    features = []
    feature_type = []
    # Features extracted using parselmouth and pratt
    f0min = 75; f0max = 500;                                                            # Limits of human speach in Hz
    pitch = call(sound_praat, "To Pitch", 0.0, f0min, f0max)                            # create a praat pitch object
    harmonicity = call(sound_praat, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)          # create a praat harmonicity object
    pointProcess = call(sound_praat, "To PointProcess (periodic, cc)", f0min, f0max)    # create a praat pointProcess object
    unit = "Hertz"
    
    features.append(call(pitch, "Get mean", 0, 0, unit)); feature_type.append(scalar);                                                        # F0 - Central Frequency
    features.append(call(pitch, "Get standard deviation", 0 ,0, unit)); feature_type.append(scalar);                                          # F0 - std
    features.append(call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)); feature_type.append(scalar);                          # Relative jitter 
    features.append(call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)); feature_type.append(scalar);                # Absolute jitter
    features.append(call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)); feature_type.append(scalar);                            # Relative average perturbation
    features.append(call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)); feature_type.append(scalar);                           # 5-point period pertubation quotient ( ppq5 )
    features.append(call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)); feature_type.append(scalar);                            # Difference of differences of periods ( ddp )
    features.append(call([sound_praat, pointProcess], "Get shimmer (local)" , 0, 0, 0.0001, 0.02, 1.3, 1.6)); feature_type.append(scalar);    # Relative Shimmer
    features.append(call([sound_praat, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)); feature_type.append(scalar);  # Relative Shimmer dB
    features.append(call([sound_praat, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)); feature_type.append(scalar);      # Shimmer (apq3)
    features.append(call([sound_praat, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)); feature_type.append(scalar);      # Shimmer (apq5)
    features.append(call([sound_praat, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)); feature_type.append(scalar);     # Shimmer (apq11)
    features.append(call([sound_praat, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)); feature_type.append(scalar);       # Shimmer (dda)
    features.append(call(harmonicity, "Get mean", 0, 0)); feature_type.append(scalar);                                                        # Harmonic Noise Ratio 
    
    
# =============================================================================
#     # Features extracted using librosa
#     features.append(librosa.feature.spectral_flatness(sound_lib)); feature_type.append(vector);           # Spectral Flatness
#     features.append(librosa.feature.rms(sound_lib)); feature_type.append(vector);                         # Volume
#     features.append(librosa.feature.zero_crossing_rate(sound_lib)); feature_type.append(vector);          # Zero Crossing Rate
#     features.append(librosa.feature.spectral_centroid(sound_lib,sr)); feature_type.append(vector);        # Spectral Centroind
#     features.append(librosa.feature.spectral_bandwidth(sound_lib,sr)); feature_type.append(vector);       # Spectral Bandwidth
#     features.append(librosa.feature.spectral_contrast(sound_lib,sr)); feature_type.append(matrix);        # Spectral Contrast
#     features.append(librosa.feature.spectral_rolloff(sound_lib,sr)); feature_type.append(vector);         # Spectral Rolloff
#     features.append(librosa.feature.mfcc(sound_lib,sr)); feature_type.append(matrix);                     # Mel-Frequency Cepstral Coefficients - MFCC 
#     features.append(librosa.feature.tonnetz(sound_lib,sr)); feature_type.append(matrix);                  # Tonnetz
#     features.append(librosa.feature.chroma_stft(sound_lib,sr)); feature_type.append(matrix);              # Spectrogram
#     features.append(librosa.feature.chroma_cqt(sound_lib,sr)); feature_type.append(matrix);               # Constant-Q Chromagram
#     features.append(librosa.feature.chroma_cens(sound_lib,sr)); feature_type.append(matrix);              # Chroma Energy Normalized
# =============================================================================
    
    # tempogram feature might be useless as it is too redundant - un comment it if you find it usefull
    #features.append(librosa.feature.tempogram(sound_lib,sr)); feature_type.append(matrix);               # Tempogram: local autocorrelation of the onset strength envelope 
    
    # Features extracted using scipy
    features.append(scipy.stats.skew(sound_lib)); feature_type.append(scalar);                            # Skewness
    # entropy = get_entropy(sound_scipy)
    # features.append(entropy[0]); feature_type.append(scalar);                                             # Entropy Left Channel
    # features.append(entropy[1]); feature_type.append(scalar);                                             # Entropy Right Channel
    # widths = np.arange(1, 31)
    # features.append(signal.cwt(sound_lib, signal.ricker, widths)); feature_type.append(matrix);           # Calculate wavelet transform using mexican hat wavelet
    # features.append(get_instantaneous_freq(sound_lib)); feature_type.append(vector);                      # Instantaneous frequency computed by hilbert transform
    
    # Features extracted by nolds (Chaos/Dynamical Systems Theory) - comment this if you didnt download nolds
    features.append(nolds.hurst_rs(sound_lib)); feature_type.append(scalar);                              # The hurst exponent is a measure of the “long-term memory” of a time series
    
    # Please dont use this even if you downloaded nolds
    # The following features require extremely long computation time and dont run by normal means, please dont use them to save yourself from having a headache. #I cant gurantite they will even coverage (dependents on the leangth of the audio file)#
    #features.append(nolds.dfa(sound_lib)); feature_type.append(vector);                                  # Performs a detrended fluctuation analysis (DFA) on the given data
    #features.append(nolds.lyap_r(sound_lib)); feature_type.append(scalar);                               # Estimates the largest Lyapunov exponent using the algorithm of Rosenstein
    #features.append(nolds.lyap_e(sound_lib)); feature_type.append(vector);                               # Estimates the Lyapunov exponents for the given data using the algorithm of Eckmann
    #features.append(nolds.corr_dim(sound_lib,1)); feature_type.append(scalar);                           # Calculates the correlation dimension with the Grassberger-Procaccia algorithm

    return features, feature_type
    
# %% Main

# Important!! read files in the following way:
def main_get_feature(directory):
    all_audio_features = []
    vowelsDict = defaultdict(list)
    for f in os.listdir(directory):
        vowel = f.split("_")[0][-1]
        if f.endswith('.wav'): # Cheak if wav file than import it otherwise continue
            data_praat = parselmouth.Sound(directory + '/' + f)
            fs_scipy, data_scipy = wavfile.read(directory + '/' + f) # Audio read by the wavfile.read function from scipy has both left channel and right channel data inside of it. Where data[:, 0] is the left channel and data[:, 1] is the right channel.
            data_librosa = librosa.load(directory + '/' + f, sr=fs_scipy)
            Features, Feature_type = get_features(data_librosa[0],data_librosa[1],data_scipy,data_praat)
            
            # Get audio features in a list choose the type features wanted, by uncommenting the relevant line
            feature_number = 1; # 0 all features, 1 scalar features, 2 vectors features, 3 matrix features
            
            if feature_number == 0: all_audio_features.append(Features) #list of all features
            if feature_number == 1: scalar_features = get_features_of_type(Features, Feature_type, scalar); all_audio_features.append(scalar_features); # list of scalar features
            if feature_number == 2: vector_features = get_features_of_type(Features, Feature_type, vector); all_audio_features.append(vector_features); # list of vector features
            if feature_number == 3: matrix_features = get_features_of_type(Features, Feature_type, matrix); all_audio_features.append(matrix_features); # list of matrix features
        
        else:
            continue
        if np.isnan(np.sum(scalar_features)):
            continue
        if '_' in f:
            vowelsDict[vowel].append(np.array(scalar_features))
        else:
            vowelsDict['iau'].append(np.array(scalar_features))
    if 'iau' in vowelsDict: del vowelsDict['iau']     
    # Normalize
    for key in vowelsDict.keys():
        tmp = np.reshape(vowelsDict[key],(len(vowelsDict[key]),len(vowelsDict[key][0])))
        vowelsDict[key] = (tmp-np.min(tmp,axis=0))/(np.max(tmp,axis=0)-np.min(tmp,axis=0))
    return vowelsDict

if __name__ == "__main__":
    main_get_feature()
    
