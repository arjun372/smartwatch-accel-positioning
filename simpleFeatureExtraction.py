import butter
from scipy.interpolate import interp1d
from scipy import signal
from scipy.fftpack import fft
from collections import Counter
import numpy as np
import shutil
import scipy
import time
import math
import os

def parseData_3DLabeled(t,x,y,z,s,Fout_Hz, kind='linear', verbose=False):
    numSamples       = len(t)
    startTime        = min(t)
    stopTime         = max(t)
    durationMs       = stopTime - startTime
    samplingRateHz   = 1000 * numSamples/float(durationMs)
    I_numSamples     = long(Fout_Hz * durationMs/1000)
    I_samplingRateHz = 1000 * I_numSamples/float(durationMs)

    max_allowable_delay_ms = (1000.0/float(Fout_Hz)) * 10.0

    # sort the arrays
    # if not assumeSorted:
    #     idx = t.argsort(kind='mergesort')
    #     t = t[idx]
    #     x = x[idx]
    #     y = y[idx]
    #     z = z[idx]
    #     s = s[idx]

    # find the time deltas
    dt = np.ediff1d(t.values)
    noticeable_lags = dt > max_allowable_delay_ms
    idx = np.argwhere(noticeable_lags).flatten()

    # set the label to -1 wherever the time deltas are greater than max_allowable_delay_ms
    for i in idx: s[i] = 0

    # interpolate the values
    I_t = np.linspace(startTime, stopTime, I_numSamples)
    I_x = interp1d(x=t, y=x, kind=kind, copy=False, assume_sorted=True)(I_t)
    I_y = interp1d(x=t, y=y, kind=kind, copy=False, assume_sorted=True)(I_t)
    I_z = interp1d(x=t, y=z, kind=kind, copy=False, assume_sorted=True)(I_t)
    I_s = interp1d(x=t, y=s, kind='zero', copy=False, assume_sorted=True)(I_t)

    if(np.size(idx) > 0) or verbose:
        print '---------------------------------------'
        print '[in]  START TIME    :', time.ctime(t[0]/1000)
        print '[in]  NUM SAMPLES   :', numSamples
        print '[in]  DURATION(ms)  :', durationMs
        print '[in]  DATA FREQ(Hz) :', samplingRateHz
        print '[out] NUM UNKNOWNS  :', np.size(idx), (np.sum(noticeable_lags)) * 100.0 / float(durationMs)
        print '[out] START TIME    :', time.ctime(I_t[0]/1000)
        print '[out] NUM SAMPLES   :', I_numSamples
        print '[out] DURATION(ms)  :', durationMs
        print '[out] DATA FREQ(Hz) :', I_samplingRateHz
        print '---------------------------------------'
    return (I_t, I_x, I_y, I_z, I_s)

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

        feature_labels.append('ln(pk)_' + label)
        features.append(np.log(pk_pwr))

        feature_labels.append('ln(energy)_' + label)
        features.append(np.log(pk_pwr_freq))

        n, min_max, mean, var, skew, kur = scipy.stats.describe(signal)
        median = np.median(signal)

        feature_labels.append('min_' + label)
        features.append(min_max[0])

        feature_labels.append('max_' + label)
        features.append(min_max[1])

        feature_labels.append('mean_' + label)
        features.append(mean)

        feature_labels.append('ln(mean_abs_dev)_' + label)
        features.append(np.log(np.mean(np.absolute(signal - mean))))

        feature_labels.append('ln(var)_' + label)
        features.append(np.log(var))

        # feature_labels.append('skew_' + label)
        # features.append(skew)
        #
        # feature_labels.append('kurtosis_' + label)
        # features.append(kur)

        feature_labels.append('ln(std_err_mean)_' + label)
        features.append(np.log(scipy.stats.sem(signal)))

        feature_labels.append('median_' + label)
        features.append(median)


    T = 1.0/samplingFreq
    N = 2**(len(signal)-1).bit_length()
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    xf[0] = xf[1]/2.0
    for signal, label in zip(f_signals, f_signal_labels):
        #n, min_max, mean, var, skew, kur = scipy.stats.describe(signal)
        entropy, pk_pwr, pk_pwr_freq = calcEntropy(Y=signal, fs=samplingFreq)

        #print median_amplitude
        #median_idx = np.where(signal == )
        #print median_idx
        # median = xf[0]

        # weakestFreq = xf[int(np.argmin(signal))]
        # feature_labels.append('min_' + label)
        # features.append(weakestFreq)
        #

        strongestFreq = xf[int(np.argmax(signal))]
        energyOfStrongest = strongestFreq * np.max(signal)

        totalEnergy = np.dot(signal, xf)
        # feature_labels.append('max_' + label)
        # features.append(strongestFreq)


        #
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
        #
        # feature_labels.append('median_' + label)
        # features.append(median)
        #
        # feature_labels.append('median_abs_dev_' + label)
        # features.append(np.median(np.absolute(signal - median)))
        #
        # feature_labels.append('median_dev_' + label)
        # features.append(np.median((signal - np.median(signal))))

        feature_labels.append('entropy_' + label)
        features.append(entropy)

        feature_labels.append('ln(pk)_' + label)
        features.append(np.log(pk_pwr))

        feature_labels.append('ln(energy)_' + label)
        features.append(np.log(pk_pwr_freq))

        feature_labels.append('ln(pwr_rat)_' + label)
        features.append(np.log(energyOfStrongest/pk_pwr_freq))

        feature_labels.append('tot_E_' + label)
        features.append(totalEnergy)

        feature_labels.append('pow_E_' + label)
        features.append(energyOfStrongest/totalEnergy)

        #print energyOfStrongest, totalEnergy, (energyOfStrongest/totalEnergy), energyOfStrongest, pk_pwr_freq, (energyOfStrongest/pk_pwr_freq)

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
        for value in singleLine[:-1]:
            tmpARFF.write(str(value) + ',')
        tmpARFF.write(str(singleLine[-1]) + '\n')
    tmpARFF.close()
    return tmpARFF

def getMajorityLabel_Nominal(data):
    assisted = ['hemi', 'four-leg-walker', 'single-leg-walker', 'crutches', 'walker']
    moving = ['running', 'walking', 'flat', 'downstairs', 'upstairs']
    stationary = ['stationary', 'sitting', 'standing', 'laying_down']
    sit_stand = ['sitting', 'standing', 'showering']
    l = data.tolist()

    # Else, check for the majority label
    l2 = []
    for s in l :
        # Return ? if even a single label here is unknown
        if s == '?':
            return '?'

        # if s in stationary:
        #     l2.append('stationary')
        #     continue

        if s == 'stationary':
            return '?'

        # if s in assisted:
        #     l2.append('assisted_walking')
        #     continue

        if s in moving or s in assisted:
            l2.append('walking')
            continue

        if s in sit_stand:
            l2.append('sit_stand')
            continue

        else:
            l2.append(s)
            continue
    val, num = Counter(l2).most_common()[0]
    return val
