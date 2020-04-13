# Voice-Disorder-SVM-Dedection
Partial implementation of the paper - ["Learning Strategies for Voice Disorder Detection"](/LearningStrategiesforVoiceDisorderDetection.pdf)

Project tested on the following DB ['Saarbruecken Voice Database'](http://stimmdb.coli.uni-saarland.de/help_en.php4)
You could also download processed DB (splitted to train & test sets) from [here](https://www.ynet.co.il/home/0,7340,L-8,00.html)

Please note that while in the original paper voice disorder anomalys were tested also on CNN and AE models, in this specific project i only tested the DB on Linear SVM model.

**Preprocessing actions-**
1.	Silence at the start and end of the audio recordings is removed (if exists).
2.	Down sampling to 16 kHz.
3.	Each file is segmented into multiple 500 ms long snippets, with a 400 ms overlap of subsequent snippets.
4.	Dataset augmentation – each sample is pitch-shifted by 8 half-semitones up and 8 half-semitones down.

**Input Representation-**

Feature based representation with **Mel-frequency cepstral coefficients** (MFCCs)

Each snippet is divided into multiple blocks for a Short-time Fourier transform (STFT) with a block size of 512 and a hop size of 128 samples, respectively. 
Then, for each block, 20 MFCCs are extracted to form a 20 * 63 input matrix and their mean values and standard deviations over blocks are calculated, resulting in a 40-dimensional feature vector per snippet.

---

### **Main Results:**

SVM detection using MFCC feature only, kernel=linear, class_weight='balanced' :

1. Max Balanced Accuracy Score of Vowel 'a' is - 53.7% , with Far = 55.77% and 18 Coefficients
2. Max Balanced Accuracy Score of Vowel 'i' is - 89.13% , with Far = 1.6% and 18 Coefficients
3. Max Balanced Accuracy Score of Vowel 'u' is - 95.35% , with Far = 5.37% and 10 Coefficients
4. Max Balanced Accuracy Score of Vowels 'aiu' is - 88.06% , with Far = 1.07% and 18 Coefficients


SVM detection using all global scalars features, kernel=linear :

1. Accuracy Score of Vowel 'a' is - 94.03% with Far = 14.81%
2. Accuracy Score of Vowel 'i' is - 81.48% with Far = 33.33%
3. Accuracy Score of Vowel 'u' is - 77.78% with Far = 92.59% 

**Global scalars features:**

Central frequency, Central frequency STD, Relative Jitter, Absolute JittSkewness, Relative Average Perturbation, 
5-point period perturbation quotient (ppq5), Difference of differences of periods (ddp), Relative Shimmer,
Relative Shimmer dB, Shimmer (apq3), Shimmer (apq5), Shimmer (apq11), Shimmer (dda), 
Harmonic noise ratio (HNR), Skewness, Hurst exponent

 ### Max Balanced Accuracy Score as function of Kernel Type
  
|          **Kernel Type**          |    Linear    |      Rbf     |     Poly     |    Sigmoid    |
|:---------------------------------:|:------------:|:------------:|:------------:|:-------------:|
|             Vowel 'a'             |    53.7%    |    74.71%    |    51.12%    |     72.96%    |
|             Vowel 'i'             |    89.13%    |    80.72%    |    52.92%    |     79.34%    |
|             Vowel 'u'             |    95.35%    |    90.08%    |    58.84%    |     85.72%    |
|            Vowels 'aiu'           |    88.06%    |     81.4%    |    71.27%    |     79.78%    |



![Image](/Balanced_accuracy_all_linear.PNG)
