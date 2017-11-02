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
import angles
import argparse
import genFeaturesOnly as gf

nFigures = 0

def plotAngles(title, t, p, r, d, l):
    global nFigures
    nFigures = nFigures + 1
    plt.figure(nFigures)
    plt.title(title)
    plt.figure(nFigures).gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.plot(t,p,'r',t,r,'g',t,d,'b', t, l, 'k.', alpha=0.6)
    plt.legend(['Pitch', 'Roll', 'Deltas'], loc='best')
    plt.grid()

def plot5D(title, t, x, y, z, a, s):
    global nFigures
    nFigures = nFigures + 1
    plt.figure(nFigures)
    plt.title(title)
    plt.figure(nFigures).gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.plot(t,x,'r',t,y,'g',t,z,'b',t,a,'k', t, s, 'k.',alpha=0.6)
    plt.legend(['X', 'Y', 'Z', 'A'], loc='best')
    plt.grid()

def plotFFT(title, xf, yf):
    global nFigures
    nFigures = nFigures + 1
    plt.figure(nFigures)
    plt.title(title)
    plt.figure(nFigures).gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.semilogy(xf*60.0, yf, 'b')
    plt.grid()

def getFs(data):
    numSamples = len(t)
    durationSec = long(max(t)-min(t))/1000
    samplingRateHz = float(numSamples/durationSec)
    print '---------------------------------------'
    print 'Plotting run : '
    print '---------------------------------------'
    print 'START TIME   :', time.ctime(min(t)/1000)
    print 'STOP  TIME   :', time.ctime(max(t)/1000)
    print 'NUM SAMPLES  :', numSamples
    print 'DURATION(s)  :', durationSec
    print 'DATA FREQ(Hz):', samplingRateHz
    return samplingRateHz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='i_file', help = "input data file location")
    parser.add_argument('-g', '--gravity-cutoff-hz', dest='gcutoff', help = "gravity low pass filter cutoff freq")
    parser.add_argument('-o', '--gravity-lp-order', dest='gorder', help = "gravity low pass filter order")
    parser.add_argument('-F', '--samplingFreq', dest='fs', help = "gravity low pass filter order")
    args = parser.parse_args()
    fs = float(args.fs)

    df = gf.importLabeledFileInterpolated(args.i_file, fs=fs, named=False,verbose=True)
    t = df.index
    x = df['aX']
    y = df['aY']
    z = df['aZ']
    l = df['class']
    a = np.hypot(x, np.hypot(y, z))

    # t = t[:100]
    # x = x[:100]
    # y = y[:100]
    # z = z[:100]
    # l = l[:100]
    # a = a[:100]


    plot5D("Raw acceleration", t, x, y, z, a, l)

    LT, LX, LY, LZ, LA = angles.linearize(t=t, x=x, y=y, z=z, cutoff=1.0, fs=fs,  order=1)
    plot5D("Linear acceleration", LT, LX, LY, LZ, LA, l)

    GT, GX, GY, GZ, GA = angles.gravity(t=t, x=x, y=y, z=z, cutoff=float(args.gcutoff), fs=fs, order=int(args.gorder))
    plot5D("Gravity from Low-Pass filter", GT, GX, GY, GZ, GA, l)

    pitch, roll = angles.pitch_roll(x, y, z)
    deltas = angles.get_angle_changes(x.tolist(), y.tolist(), z.tolist())
    plotAngles("Pitch & Roll & Angle Deltas", GT, pitch, roll, deltas, l)

    plt.show()
