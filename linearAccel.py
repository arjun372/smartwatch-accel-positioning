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
    plt.plot(t,x,'r',t,y,'g',t,z,'b',t,a,'k', t, s, 'c',alpha=0.6)
    plt.legend(['X', 'Y', 'Z', 'A', 'A2'], loc='best')
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
    a = df['aA']
    l = df['class']

    # window = int(fs*10)
    # t = t[:window]
    # x = x[:window]
    # y = y[:window]
    # z = z[:window]
    # l = l[:window]
    # a = a[:window]


    plot5D("Raw acceleration", t, x, y, z, a, l)

    LX, LY, LZ, LA, LA2 = angles.linearize(x=x, y=y, z=z, a=a, cutoff=1.0, fs=fs,  order=1)
    plot5D("Linear acceleration", t, LX, LY, LZ, LA, LA2)

    GX, GY, GZ, GA, GA2 = angles.gravity(x=x, y=y, z=z, a=a, cutoff=float(args.gcutoff), fs=fs, order=int(args.gorder))
    plot5D("Gravity from Low-Pass filter", t, GX, GY, GZ, GA, GA2)

    pitch, roll = angles.pitch_roll(x, y, z)
    deltas = angles.get_angle_changes(x.tolist(), y.tolist(), z.tolist(), a.tolist())
    plotAngles("Pitch & Roll & Angle Deltas", t, pitch, roll, deltas, l)

    #for val in deltas: print val
    Gpitch, Groll = angles.pitch_roll(GX, GY, GZ)
    Gdeltas = angles.get_angle_changes(GX, GY, GZ, GA2)
    plotAngles("Low-Passed - Pitch & Roll & Angle Deltas", t, Gpitch, Groll, Gdeltas, l)

    plt.show()
