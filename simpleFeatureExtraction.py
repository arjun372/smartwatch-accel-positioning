import butter
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from collections import deque
from termcolor import colored
from itertools import islice
from scipy import signal
import multiprocessing
from scipy.fftpack import fft
from collections import Counter
from scipy.stats import mode
import pandas as pd
import numpy as np
import tempfile
import shutil
import scipy
import time
import math
import time
import os

import weka.core.jvm as jvm
import weka.classifiers as weka_clf
from weka.classifiers import Evaluation
from weka.core.converters import Loader
import weka.core.serialization as serialization

POS_PATH = 'bin'

TIME_DOMAIN = ["aX_t", "aY_t", "aZ_t", "aA_t", "gX_t", "gY_t", "gZ_t", "gA_t", "LP_aX_t", "LP_aY_t", "LP_aZ_t", "LP_aA_t", "LP_gX_t", "LP_gY_t", "LP_gZ_t", "LP_gA_t"]

GRAVITY   = 9.79607
clf_started = False

def getMag(a):
            return math.sqrt(sum(i**2 for i in a))

def getAbs(a):
    return abs(a)

def getAmplitude(x,y,z):
    lenx = len(x)
    gravity = 9.79607
    a = np.zeros(lenx)
    for i in xrange(lenx):
        print x[i]
        a[i] = (math.sqrt((x[i]**2) + (y[i]**2) + (z[i]**2)) - gravity)
    return a

def sliding_window(iterable, size=2, step=1, fillvalue=None):
    if size < 0 or step < 1:
        raise ValueError
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        try:
            q.append(next(it))
        except StopIteration: # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))

def sortTogether(idx, vals):
    sorted_together =  sorted(zip(idx, vals))
    sorted_idx = [x[0] for x in sorted_together]
    sorted_vals = [x[1] for x in sorted_together]
    return sorted_idx, sorted_vals

def parseData_1D(iFile, Fout_Hz=16):
    data = iFile.split(',')
    t,x = sortTogether(idx=map(long, data[::2]), vals=map(float, data[1::2]))
    numSamples       = len(t)
    startTime        = min(t)
    stopTime         = max(t)
    durationMs       = stopTime - startTime
    samplingRateHz   = 1000 * numSamples/float(durationMs)
    I_numSamples     = long(Fout_Hz * durationMs/1000)
    I_samplingRateHz = 1000 * I_numSamples/float(durationMs)
    I_t              = np.linspace(startTime, stopTime, I_numSamples)
    I_x              = interp1d(t, x)(I_t)
    # print '---------------------------------------'
    # print '[in]  START TIME   :', time.ctime(t[0]/1000)
    # print '[in]  NUM SAMPLES  :', numSamples
    # print '[in]  DURATION(ms)  :', durationMs
    # print '[in]  DATA FREQ(Hz):', samplingRateHz
    # print '[out] START TIME   :', time.ctime(I_t[0]/1000)
    # print '[out] NUM SAMPLES  :', I_numSamples
    # print '[out] DURATION(ms)  :', durationMs
    # print '[out] DATA FREQ(Hz):', I_samplingRateHz
    return (I_x)

def parseData_3DLabeled(t,x,y,z,s,Fout_Hz, kind='linear'):
    numSamples       = len(t)
    startTime        = min(t)
    stopTime         = max(t)
    durationMs       = stopTime - startTime
    samplingRateHz   = 1000 * numSamples/float(durationMs)
    I_numSamples     = long(Fout_Hz * durationMs/1000)
    I_samplingRateHz = 1000 * I_numSamples/float(durationMs)
    print '---------------------------------------'
    print '[in]  START TIME    :', time.ctime(t[0]/1000)
    print '[in]  NUM SAMPLES   :', numSamples
    print '[in]  DURATION(ms)  :', durationMs
    print '[in]  DATA FREQ(Hz) :', samplingRateHz

    max_allowable_delay_ms = (1000.0/float(Fout_Hz)) * 10.0

    # sort the arrays
    # idx = t.argsort(kind='mergesort')
    # t = t[idx]
    # #print x[t]
    # x = x[t]
    # y = y[t]
    # z = z[t]
    # s = s[t]

    # find the time deltas
    dt = np.ediff1d(t.values)
    noticeable_lags = dt > max_allowable_delay_ms
    idx = np.argwhere(noticeable_lags).flatten()
    # set the label to -1 wherever the time deltas are greater than max_allowable_delay_ms
    for i in idx:
        s[i] = 0

    # interpolate the values
    I_t = np.linspace(startTime, stopTime, I_numSamples)
    I_x = interp1d(x=t, y=x, kind=kind, copy=False, assume_sorted=True)(I_t)
    I_y = interp1d(x=t, y=y, kind=kind, copy=False, assume_sorted=True)(I_t)
    I_z = interp1d(x=t, y=z, kind=kind, copy=False, assume_sorted=True)(I_t)
    I_s = interp1d(x=t, y=s, kind='zero', copy=False, assume_sorted=True)(I_t)
    I_a = np.hypot(I_x, np.hypot(I_y, I_z))

    print '[out] NUM UNKNOWNS  :', np.size(idx), (np.sum(noticeable_lags)) * 100.0 / float(durationMs)
    print '[out] START TIME    :', time.ctime(I_t[0]/1000)
    print '[out] NUM SAMPLES   :', I_numSamples
    print '[out] DURATION(ms)  :', durationMs
    print '[out] DATA FREQ(Hz) :', I_samplingRateHz
    print '---------------------------------------'
    return (I_t, I_x, I_y, I_z, I_s)

def parseData_3D(t,x,y,z, Fout_Hz):
    numSamples       = len(t)
    startTime        = min(t)
    stopTime         = max(t)
    durationMs       = stopTime - startTime
    samplingRateHz   = 1000 * numSamples/float(durationMs)
    I_numSamples     = long(Fout_Hz * durationMs/1000)
    I_samplingRateHz = 1000 * I_numSamples/float(durationMs)
    print '---------------------------------------'
    print '[in]  START TIME    :', time.ctime(t[0]/1000)
    print '[in]  NUM SAMPLES   :', numSamples
    print '[in]  DURATION(ms)  :', durationMs
    print '[in]  DATA FREQ(Hz) :', samplingRateHz

    if(samplingRateHz < 1):
        print "Sampling Rate is too low for accurate prediction, quitting..."
        exit(1)

    I_t  = np.linspace(startTime, stopTime, I_numSamples)
    I_x  = interp1d(t, x)(I_t)
    I_y  = interp1d(t, y)(I_t)
    I_z  = interp1d(t, z)(I_t)
    print '[out] START TIME    :', time.ctime(I_t[0]/1000)
    print '[out] NUM SAMPLES   :', I_numSamples
    print '[out] DURATION(ms)  :', durationMs
    print '[out] DATA FREQ(Hz) :', I_samplingRateHz
    print '---------------------------------------'
    return (I_t, I_x, I_y, I_z)

def parseData_4D(t,x,y,z,a, Fout_Hz):
    numSamples       = len(t)
    startTime        = min(t)
    stopTime         = max(t)
    durationMs       = stopTime - startTime
    samplingRateHz   = 1000 * numSamples/float(durationMs)
    I_numSamples     = long(Fout_Hz * durationMs/1000)
    I_samplingRateHz = 1000 * I_numSamples/float(durationMs)
    print '---------------------------------------'
    print '[in]  START TIME    :', time.ctime(t[0]/1000)
    print '[in]  NUM SAMPLES   :', numSamples
    print '[in]  DURATION(ms)  :', durationMs
    print '[in]  DATA FREQ(Hz) :', samplingRateHz
    print '[out] START TIME    :', time.ctime(I_t[0]/1000)
    print '[out] NUM SAMPLES   :', I_numSamples
    print '[out] DURATION(ms)  :', durationMs
    print '[out] DATA FREQ(Hz) :', I_samplingRateHz
    I_t              = np.linspace(startTime, stopTime, I_numSamples)
    I_x              = interp1d(t, x)(I_t)
    I_y              = interp1d(t, y)(I_t)
    I_z              = interp1d(t, z)(I_t)
    I_a              = interp1d(t, a)(I_t)
    return (I_x, I_y, I_z, I_a)

def getFFT2(y, fs):
    N = 2**(len(y)-1).bit_length()
    #if(len(y) != N):
        #print 'FFT using N pts:', N, len(y)
    T = 1.0 / float(fs)
    yf = fft(x=y*np.hanning(len(y)), n=N)
    return (2.0/N * np.abs(yf[0:N//2]))


def calcEntropy(Y, fs):
    N = 2**(len(Y)-1).bit_length()
    if(len(Y) != N):
        #print 'Entropy using N pts:', N, len(Y)
        Y = np.append(Y, np.zeros(N-len(Y)))
    freq, Pxx_den_k = scipy.signal.welch(Y, fs, nperseg=N, scaling='density')

    # Calculate entropy
    norm_Pxx_den_k = Pxx_den_k / sum(Pxx_den_k)
    entropy = -1.0 * sum(norm_Pxx_den_k * np.log(norm_Pxx_den_k))
    energy = sum(Pxx_den_k * freq)
    pk_pwr = np.max(Pxx_den_k)
    return entropy, pk_pwr, energy

# Returns the index of the second harmonic, along with its amplitude, bonus sum of energy of all peaks
def get2ndHarmonic(Y,fs):
    T = 1.0 / fs
    Y2 = np.delete(Y, np.argmax(Y))
    idx2 = signal.find_peaks_cwt(Y, np.arange(1, 1/(2*T)))
    return np.where(Y==max(Y2[idx2])), max(Y2)

def getLessFeatures(name, signal, samplingFreq):

    t_signal_labels = [name]
    t_signals       = [signal]

    f_signal_labels = []
    f_signals       = []

    # perform FFT
    for signal, label in zip(t_signals, t_signal_labels):
        FFT = getFFT2(y=signal, fs=samplingFreq)
        f_signal_labels.append('FFT_' + label)
        f_signals.append(FFT)

    # generate features
    feature_labels = []
    features       = []

    for signal, label in zip(t_signals, t_signal_labels):
        entropy, pk_pwr, pk_pwr_freq = calcEntropy(Y=signal, fs=samplingFreq)

        feature_labels.append('entropy_' + label)
        features.append(entropy)

        feature_labels.append('pk_' + label)
        features.append(pk_pwr)

        feature_labels.append('energy_' + label)
        features.append(pk_pwr_freq)

        n, min_max, mean, var, skew, kur = scipy.stats.describe(signal)
        median = np.median(signal)

        feature_labels.append('min_' + label)
        features.append(min_max[0])

        feature_labels.append('max_' + label)
        features.append(min_max[1])

        feature_labels.append('mean_' + label)
        features.append(mean)

        feature_labels.append('mean_abs_dev_' + label)
        features.append(np.mean(np.absolute(signal - mean)))

        # feature_labels.append('variance_' + label)
        # features.append(var)
        #
        # feature_labels.append('skew_' + label)
        # features.append(skew)
        #
        # feature_labels.append('kurtosis_' + label)
        # features.append(kur)
        #
        # feature_labels.append('std_err_mean_' + label)
        # features.append(scipy.stats.sem(signal))

        feature_labels.append('median_' + label)
        features.append(median)


    T = 1.0/samplingFreq
    N = len(signal)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

    for signal, label in zip(f_signals, f_signal_labels):
        #n, min_max, mean, var, skew, kur = scipy.stats.describe(signal)
        entropy, pk_pwr, pk_pwr_freq = calcEntropy(Y=signal, fs=samplingFreq)

        #print median_amplitude
        #median_idx = np.where(signal == )
        #print median_idx
        # median = xf[0]
        # weakestFreq = np.min(signal)[0]#xf[int(np.argmin(signal))]
        # strongestFreq = xf[int(np.argmax(signal))]
        #
        # feature_labels.append('min_' + label)
        # features.append(weakestFreq)
        #
        # feature_labels.append('max_' + label)
        # features.append(strongestFreq)

        # feature_labels.append('mean_' + label)
        # features.append(mean)
        #
        # feature_labels.append('mean_abs_dev_' + label)
        # features.append(np.mean(np.absolute(signal - mean)))
        #
        # feature_labels.append('variance_' + label)
        # features.append(var)
        #
        # feature_labels.append('skew_' + label)
        # features.append(skew)
        #
        # feature_labels.append('kurtosis_' + label)
        # features.append(kur)
        #
        # feature_labels.append('std_err_mean_' + label)
        # features.append(scipy.stats.sem(signal))

        # feature_labels.append('median_' + label)
        # features.append(median)

        # feature_labels.append('median_abs_dev_' + label)
        # features.append(np.median(np.absolute(signal - median)))

        # feature_labels.append('median_dev_' + label)
        # features.append(np.median((signal - np.median(signal))))

        feature_labels.append('entropy_' + label)
        features.append(entropy)

        feature_labels.append('pk_' + label)
        features.append(pk_pwr)

        feature_labels.append('energy_' + label)
        features.append(pk_pwr_freq)

    return [f for f in feature_labels], features

def writeARFF(df, name):
    cols = df.columns.tolist()
    class_idx = cols.index('class')
    cols.append(cols.pop(class_idx))
    attrs = df[cols].values.tolist()
    labels = cols[:-1]
    nominal=list(sorted(set(df[cols[-1]])))
    # Write features to a temp rawInputFile
    tmpARFF = open(name,"w")
    tmpARFF.write('@relation '+name+'\n')
    for label in labels:tmpARFF.write('@attribute '+label+' numeric\n')
    tmpARFF.write('@attribute class {')
    for x in xrange(0, len(nominal)-1):tmpARFF.write(nominal[x]+',')
    tmpARFF.write(nominal[-1]+'}\n')
    tmpARFF.write('@data ')
    for singleLine in attrs:
        for value in singleLine:tmpARFF.write(str(value)+',')
        tmpARFF.write('?\n')
    tmpARFF.close()
    return tmpARFF

def getMajorityLabel_Nominal(data):
    moving = ['running', 'walking', 'hemi', 'four-leg-walker', 'single-leg-walker', 'crutches', 'walker', 'flat', 'downstairs', 'upstairs']
    stationary = ['stationary', 'sitting', 'standing', 'laying_down']
    l = data.tolist()

    # Return ? if even a single label here is unknown
    if '?' in l:
        return '?'

    # Else, check for the majority label
    l2 = []
    for s in l :
        if s in stationary:
            l2.append('stationary')
            continue
        #if s == 'stationary':
        #    l2.append('?')
        #    continue
        elif s in moving:
            l2.append('moving')
            continue
        else:
            l2.append(s)
            continue
    val, num = Counter(l2).most_common()[0]
    return val
