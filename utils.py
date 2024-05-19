import pandas as pd
import joblib
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
import numpy as np 
import librosa as lb
import math
from scipy import stats
import wave

sc_X = StandardScaler()

#--------------------
# DF para ejemplo
#--------------------
'''
df = pd.read_csv('MFCC.csv')

sample = df.sample(1)
'''
#--------------------
# Modelos
#--------------------

LRm, ref_cols, target = joblib.load('LR.pkl')
SVMCm, ref_cols, target = joblib.load('SVMC.pkl')
SVMC1m, ref_cols, target = joblib.load('SVMC1.pkl')
RFCm, ref_cols, target = joblib.load('RFC.pkl')
KNNCm, ref_cols, target = joblib.load('KNNC.pkl')

#--------------------
# Funcion votos
#--------------------

def voting(dataF):
    X_ = dataF

    predicts = []
    models = [LRm, RFCm, KNNCm, SVMCm, SVMC1m]

    for item in models:
        predicts.append(item.predict(X_)[0])
        print(item)
        print(item.predict(X_)[0])

    
    mean = (predicts[0]+predicts[1]+predicts[2]+predicts[3]+predicts[4])/5

    if mean > 0.65:
        return 1
    else:
        return 0

#--------------------
# Funcion m4a a wav
#--------------------

def convertidor(m4a_file):
    wav_filename = 'output.wav'
    sound = AudioSegment.from_file(m4a_file, format = 'mp4')
    file_handle = sound.export(wav_filename, format = 'wav')

    return file_handle

#--------------------
# Rene + Memo
#--------------------


#Mean x

def mean_x(matrix): 
    matrix_z = []
    matrix_t = matrix.T

    for i in range(matrix_t.shape[0]): 
        matrix_z.append(list(stats.zscore(matrix_t[i])))

    matrix_z = np.array(matrix_z)

    matrix = matrix_z.T 

    mean_arr = np.mean(matrix.T, axis=0)

    return (mean_arr, matrix)

def std_x(arr): 
    #STD arr
    std_arr = []
    for i in range(arr.shape[0]):
        std_arr.append(np.std(arr[i])) 

    std_arr = np.array(std_arr)
    return std_arr

import wave

def bytes_to_wav(byte_data, filename):
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(byte_data)


def to_diagnose(file):
    #Read file
    signal, sr = lb.load(file)

    #MFCCs & DeltaS
    mfccs = lb.feature.mfcc(y = signal, n_mfcc=13, sr=sr)

    delta_mfccs = lb.feature.delta(mfccs) #Get delta
    delta_delta_mfccs = lb.feature.delta(mfccs, order=2) #Get delta delta
    
    #Mean coef 
    mean_coef, mfccs_z = mean_x(mfccs)

    #Mean Delta 
    mean_delta, delta_mfccs_z = mean_x(delta_mfccs)

    #Mean DeltaDelta
    mean_delta2, delta_2_mfccs_z = mean_x(delta_delta_mfccs)

    #STD Coef 
    std_coef = std_x(mfccs_z)

    #STD Delta 
    std_delta = std_x(delta_mfccs_z)

    #STD Delta 
    std_delta2 = std_x(delta_2_mfccs_z)

    d = [{
        'mean_MFCC_0th_coef': mean_coef[0],
        'mean_MFCC_1st_coef': mean_coef[1],
        'mean_MFCC_2nd_coef': mean_coef[2],
        'mean_MFCC_3rd_coef': mean_coef[3],
        'mean_MFCC_4th_coef': mean_coef[4],
        'mean_MFCC_5th_coef': mean_coef[5],
        'mean_MFCC_6th_coef': mean_coef[6],
        'mean_MFCC_7th_coef': mean_coef[7],
        'mean_MFCC_8th_coef': mean_coef[8],
        'mean_MFCC_9th_coef': mean_coef[9],
        'mean_MFCC_10th_coef': mean_coef[10],
        'mean_MFCC_11th_coef': mean_coef[11],
        'mean_MFCC_12th_coef': mean_coef[12],
        'mean_0th_delta': mean_delta[0],
        'mean_1st_delta': mean_delta[1],
        'mean_2nd_delta': mean_delta[2],
        'mean_3rd_delta': mean_delta[3],
        'mean_4th_delta': mean_delta[4],
        'mean_5th_delta': mean_delta[5],
        'mean_6th_delta': mean_delta[6],
        'mean_7th_delta': mean_delta[7],
        'mean_8th_delta': mean_delta[8],
        'mean_9th_delta': mean_delta[9],
        'mean_10th_delta': mean_delta[10],
        'mean_11th_delta': mean_delta[11],
        'mean_12th_delta': mean_delta[12],
        'mean_delta_delta_0th': mean_delta2[0],
        'mean_1st_delta_delta': mean_delta2[1],
        'mean_2nd_delta_delta': mean_delta2[2],
        'mean_3rd_delta_delta': mean_delta2[3],
        'mean_4th_delta_delta': mean_delta2[4],
        'mean_5th_delta_delta': mean_delta2[5],
        'mean_6th_delta_delta': mean_delta2[6],
        'mean_7th_delta_delta': mean_delta2[7],
        'mean_8th_delta_delta': mean_delta2[8],
        'mean_9th_delta_delta': mean_delta2[9],
        'mean_10th_delta_delta': mean_delta2[10],
        'mean_11th_delta_delta': mean_delta2[11],
        'mean_12th_delta_delta': mean_delta2[12],
        'std_MFCC_0th_coef': std_coef[0],
        'std_MFCC_1st_coef': std_coef[1],
        'std_MFCC_2nd_coef': std_coef[2],
        'std_MFCC_3rd_coef': std_coef[3],
        'std_MFCC_4th_coef': std_coef[4],
        'std_MFCC_5th_coef': std_coef[5],
        'std_MFCC_6th_coef': std_coef[6],
        'std_MFCC_7th_coef': std_coef[7],
        'std_MFCC_8th_coef': std_coef[8],
        'std_MFCC_9th_coef': std_coef[9],
        'std_MFCC_10th_coef': std_coef[10],
        'std_MFCC_11th_coef': std_coef[11],
        'std_MFCC_12th_coef': std_coef[12],
        'std_0th_delta': std_delta[0],
        'std_1st_delta': std_delta[1],
        'std_2nd_delta': std_delta[2],
        'std_3rd_delta': std_delta[3],
        'std_4th_delta': std_delta[4],
        'std_5th_delta': std_delta[5],
        'std_6th_delta': std_delta[6],
        'std_7th_delta': std_delta[7],
        'std_8th_delta': std_delta[8],
        'std_9th_delta': std_delta[9],
        'std_10th_delta': std_delta[10],
        'std_11th_delta': std_delta[11],
        'std_12th_delta': std_delta[12],
        'std_delta_delta_0th': std_delta2[0],
        'std_1st_delta_delta': std_delta2[1],
        'std_2nd_delta_delta': std_delta2[2],
        'std_3rd_delta_delta': std_delta2[3],
        'std_4th_delta_delta': std_delta2[4],
        'std_5th_delta_delta': std_delta2[5],
        'std_6th_delta_delta': std_delta2[6],
        'std_7th_delta_delta': std_delta2[7],
        'std_8th_delta_delta': std_delta2[8],
        'std_9th_delta_delta': std_delta2[9],
        'std_10th_delta_delta': std_delta2[10],
        'std_11th_delta_delta': std_delta2[11],
        'std_12th_delta_delta': std_delta2[12]
    }]

    df = pd.DataFrame.from_dict(d)
    df = df.iloc[:, 5:]

    return df 

#--------------------
'''
X_new = sample[ref_cols]




print(voting(sample))
'''

