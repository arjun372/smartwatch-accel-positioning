import sys
import time
import numpy as np
import math
import itertools
from scipy import stats
from scipy import signal
import butter as butter
from scipy.fftpack import fft
from scipy.interpolate import interp1d
import pandas as pd

def gravity(x, y, z, a, cutoff, fs, order):
    X = butter.lowpass_filter(data=x, cutoff=cutoff, fs=fs, order=order)
    Y = butter.lowpass_filter(data=y, cutoff=cutoff, fs=fs, order=order)
    Z = butter.lowpass_filter(data=z, cutoff=cutoff, fs=fs, order=order)
    A = butter.lowpass_filter(data=a, cutoff=cutoff, fs=fs, order=order)
    A2 = np.hypot(X, np.hypot(Y, Z))
    return X, Y, Z, A, A2

def linearize(x, y, z, a, cutoff, fs, order):
    X = butter.highpass_filter(data=x, cutoff=cutoff, fs=fs, order=order)
    Y = butter.highpass_filter(data=y, cutoff=cutoff, fs=fs, order=order)
    Z = butter.highpass_filter(data=z, cutoff=cutoff, fs=fs, order=order)
    A = butter.highpass_filter(data=a, cutoff=cutoff, fs=fs, order=order)
    A2 = np.hypot(X, np.hypot(Y, Z))
    return X, Y, Z, A, A2


def get_angle_changes(x, y, z, a):
    xp = x[0]
    yp = y[0]
    zp = z[0]
    ap = a[0]
    angles = np.zeros(len(x))
    for i in xrange(0, len(x)):
        #denominator
        den = a[i] * ap
        ap = a[i]
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
    pitch = np.rad2deg(np.arctan2(y, z))
    roll = np.rad2deg(np.arctan2(-1.0 * x, np.hypot(y, z)))
    return pitch, roll
