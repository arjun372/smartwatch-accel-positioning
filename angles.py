import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
from scipy import stats
from scipy import signal
import butter as butter
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
