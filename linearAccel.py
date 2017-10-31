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

nFigures = 0

def getFFT(y, fs):
    N = 2**(len(y)-1).bit_length()
    print 'FFT using N pts:', N
    T = 1.0 / float(fs)
    yf = fft(x=y, n=N)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    return xf, (2.0/N * np.abs(yf[0:N//2]))

def interpolate(t, x,y,z, fo=16.0, order='zero'):
    numSamples = len(t)
    durationSec = long(max(t)-min(t))/1000
    samplingRateHz = float(numSamples/(float)(durationSec))
    I_numSamples     = long(fo * durationSec)
    I_samplingRateHz = 1000 * I_numSamples/float(durationSec)
    I_t              = np.linspace(min(t), max(t), I_numSamples)
    X = interp1d(t, x, kind=order, copy=False, assume_sorted=False)(I_t)
    Y = interp1d(t, y, kind=order, copy=False, assume_sorted=False)(I_t)
    Z = interp1d(t, z, kind=order, copy=False, assume_sorted=False)(I_t)
    return I_t, X, Y, Z

def readCSV(filename, decimals=3, median=-1):
    i_file = open(filename).readlines()
    t = np.array([long (i.split(',')[0]) for i in i_file])
    x = np.around(np.array([float(i.split(',')[1]) for i in i_file]), decimals=decimals)
    y = np.around(np.array([float(i.split(',')[2]) for i in i_file]), decimals=decimals)
    z = np.around(np.array([float(i.split(',')[3]) for i in i_file]), decimals=decimals)
    s = np.array([str(i.split(',')[4]) for i in i_file])
    if(median >= 1):
        x = signal.medfilt(x, median)
        y = signal.medfilt(y, median)
        z = signal.medfilt(z, median)
    return t, x, y, z, s

def plotAngles(title, t, p, r, d):
    global nFigures
    nFigures = nFigures + 1
    plt.figure(nFigures)
    plt.title(title)
    plt.figure(nFigures).gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.plot(t,p,'r',t,r,'g',t,d,'b', alpha=0.6)
    plt.legend(['Pitch', 'Roll', 'Deltas'], loc='best')
    plt.grid()

def plot4D(title, t, x, y, z, a):
    global nFigures
    nFigures = nFigures + 1
    plt.figure(nFigures)
    plt.title(title)
    plt.figure(nFigures).gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.plot(t,x,'r',t,y,'g',t,z,'b',t,a,'k.',alpha=0.6)
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
    args = parser.parse_args()
    fs = 10.0
    t, x, y, z, s = readCSV(args.i_file)
    t, x, y, z, s = fe.parseData_3DLabeled(t=t,x=x,y=y,z=z,s=s,Fout_Hz=fs)
    #t, x, y, z, a = angles.gravity(t=t, x=x, y=y, z=z, cutoff=5.0, fs=fs, order=1)
    a = np.hypot(x, np.hypot(y, z))
    fs = getFs(t)

    # t = t[:100]
    # x = x[:100]
    # y = y[:100]
    # z = z[:100]
    # a = a[:100]

    plot4D("Raw acceleration", t, x, y, z, s)

    LT, LX, LY, LZ, LA = angles.linearize(t=t, x=x, y=y, z=z, cutoff=1.0, fs=fs,  order=1)
    plot4D("Linear acceleration", LT, LX, LY, LZ, LA)

    GT, GX, GY, GZ, GA = angles.gravity(t=t, x=x, y=y, z=z, cutoff=float(args.gcutoff), fs=fs, order=int(args.gorder))
    plot4D("Gravity from Low-Pass filter", GT, GX, GY, GZ, GA)

    pitch, roll = angles.pitch_roll(GX,GY,GZ)
    deltas = angles.get_angle_changes(GX,GY,GZ)
    plotAngles("Pitch & Roll & Angle Deltas", GT, pitch, roll, deltas)

    plt.show()
