import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
from scipy import stats
from scipy import signal
import butter as butter
import simpleFeatureExtraction as fe
from scipy.fftpack import fft
from scipy.interpolate import interp1d
import pandas as pd

def gravity(t, x, y, z, cutoff, fs, order):
    X = butter.lowpass_filter(data=x, cutoff=cutoff, fs=fs, order=order)
    Y = butter.lowpass_filter(data=y, cutoff=cutoff, fs=fs, order=order)
    Z = butter.lowpass_filter(data=z, cutoff=cutoff, fs=fs, order=order)
    A = np.hypot(X, np.hypot(Y, Z))
    pad = 0#int(1.0/cutoff * fs)
    return t[pad:], X[pad:], Y[pad:], Z[pad:], A[pad:]

def linearize(t, x, y, z, cutoff, fs, order):
    X = butter.highpass_filter(data=x, cutoff=cutoff, fs=fs, order=order)
    Y = butter.highpass_filter(data=y, cutoff=cutoff, fs=fs, order=order)
    Z = butter.highpass_filter(data=z, cutoff=cutoff, fs=fs, order=order)
    A = np.hypot(X, np.hypot(Y, Z))
    pad = 0#int(1.0/cutoff * fs)
    return t[pad:], X[pad:], Y[pad:], Z[pad:], A[pad:]


def get_angle_changes(x, y, z):
    xp = 0.0
    yp = 0.0
    zp = 1.0
    ap = 1.0
    angles = np.zeros(len(x))
    for i in xrange(0, len(x)):
        #denominator
        a_i = np.hypot(x[i], np.hypot(y[i], z[i]))
        den = a_i * ap
        ap = a_i
        # numerator
        num = x[i] * xp + y[i] * yp + z[i] * zp
        xp = x[i]
        yp = y[i]
        zp = z[i]
        # angle
        angles[i] = np.arccos(num/den)
        if(np.isnan(angles[i])):
            angles[i] = 0.0
    angles[0] = angles[1]
    return angles

def pitch_roll(x, y, z):
    pitch = np.arctan2(y, z)
    roll = np.arctan2(-1.0 * x, np.hypot(y, z))
    return pitch, roll

def getLessFeatures_pandas(df, Fs, N):
    idx = [(df.index)[0]]

    if(len(df) < int(N * Fs)):
        return pd.DataFrame(index=idx)

    label = fe.getMajorityLabel_Nominal(df['class'])

    if label == '?':
        return pd.DataFrame(index=idx)

    t = df.index
    x = df['aX_t']
    y = df['aY_t']
    z = df['aZ_t']
    # x = np.around(df['aX_t'], 1)
    # y = np.around(df['aY_t'], 1)
    # z = np.around(df['aZ_t'], 1)
    # x = signal.medfilt(df['aX_t'], 7)
    # y = signal.medfilt(df['aY_t'], 7)
    # z = signal.medfilt(df['aZ_t'], 7)

    LT, LX, LY, LZ, LA = linearize(t=t, x=x, y=y, z=z, cutoff=1.0, fs=Fs, order=1)
    GT, GX, GY, GZ, GA = gravity(t=t, x=x, y=y, z=z, cutoff=0.25, fs=Fs, order=1)

    pitch, roll = pitch_roll(GX,GY,GZ)
    deltas = get_angle_changes(GX,GY,GZ)

    GXL, GXF = fe.getLessFeatures(name="GX", signal=GX, samplingFreq=Fs)
    GYL, GYF = fe.getLessFeatures(name="GY", signal=GY, samplingFreq=Fs)
    GZL, GZF = fe.getLessFeatures(name="GZ", signal=GZ, samplingFreq=Fs)
    GAL, GAF = fe.getLessFeatures(name="GA", signal=GA, samplingFreq=Fs)

    LXL, LXF = fe.getLessFeatures(name="LX", signal=LX, samplingFreq=Fs)
    LYL, LYF = fe.getLessFeatures(name="LY", signal=LY, samplingFreq=Fs)
    LZL, LZF = fe.getLessFeatures(name="LZ", signal=LZ, samplingFreq=Fs)
    LAL, LAF = fe.getLessFeatures(name="LA", signal=LA, samplingFreq=Fs)

    PL, PF = fe.getLessFeatures(name="PTCH", signal=pitch,  samplingFreq=Fs)
    RL, RF = fe.getLessFeatures(name="ROLL", signal=roll,   samplingFreq=Fs)
    DL, DF = fe.getLessFeatures(name="DELT", signal=deltas, samplingFreq=Fs)

    data = pd.DataFrame([GXF + GYF + GZF + GAF + LXF + LYF + LZF + LAF + PF + RF + DF + [label]], columns=(GXL + GYL + GZL + GAL + LXL + LYL + LZL + LAL + PL + RL + DL + ['class']), index=idx)
    return data
