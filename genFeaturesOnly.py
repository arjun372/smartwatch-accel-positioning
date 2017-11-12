from joblib import Parallel, delayed
import simpleFeatureExtraction as fe
import multiprocessing
import pandas as pd
import numpy as np
import tempfile
import fnmatch
import scipy
import glob
import argparse
import math
import angles as angles
import os

labeled_data_dir  = glob.glob('data/train/*.csv')
test_data_dir     = glob.glob('data/test/*.csv')
feature_data_dir  = 'data/generated_features/'
num_cores         = multiprocessing.cpu_count()

A_THRES = 50
G_THRES = 50

LABEL_COLUMN = 'class'
INDEX_COLUMN = "t"

ACCL = ["aX", "aY", "aZ"]
GYRO = ["gX", "gY", "gZ"]

CSV_COL_NAMES = [INDEX_COLUMN] + ACCL + GYRO + [LABEL_COLUMN]

def importCSV(srcDir, recursive):
    if(recursive == False):return glob.glob(srcDir+'/*.csv')
    matches = []
    for root, dirnames, filenames in os.walk(srcDir):
        for filename in fnmatch.filter(filenames, '*.csv'):
            matches.append(os.path.join(root, filename))
    return matches

# Imports the CSV file as a dataframe
def importLabeledFile(file_):

    # Read CSV and sort it based on it's timestamp value
    df = pd.read_csv(file_, header=None, names=CSV_COL_NAMES, index_col=INDEX_COLUMN)
    df.sort_index(inplace=True)

    # removes all accel values greater than the threshold.
    # Needed since some sony watches output garbage at times
    df = df[(df[ACCL[0]] < A_THRES) & (df[ACCL[1]] < A_THRES) & (df[ACCL[2]] < A_THRES)]

    unlabeled3D = (df[GYRO[0]].isnull().values.all())
    if(unlabeled3D):
        print 'Unlabeled 3D file :', file_
        df.drop(GYRO, axis=1, inplace=True)
        df[LABEL_COLUMN] = '?'
        return df

    labeled3D = not unlabeled3D and (df[GYRO[2]].isnull().values.all())
    if(labeled3D):
        print 'Labeled 3D file :', file_
        df.rename(columns={GYRO[0]: LABEL_COLUMN, LABEL_COLUMN: GYRO[0]}, inplace=True)
        df.drop(GYRO, axis=1, inplace=True)
        return df


    labeled6D = not labeled3D and not (df[LABEL_COLUMN].isnull().values.all())
    if(labeled6D):
        print 'Labeled 6D file :', file_
        df.drop(GYRO, axis=1, inplace=True)
        return df

    unlabeled6D = not labeled6D
    print 'Unlabeled 6D file :', file_
    df.drop(GYRO, axis=1, inplace=True)
    df[LABEL_COLUMN] = '?'

    return df
def importLabeledFileInterpolated(file_, fs, named=True, verbose=False):
    # import the file
    df = importLabeledFile(file_)

    # add all labels, including unknown '?'
    labels = list(set(df[LABEL_COLUMN]))
    labels.insert(0, '?')

    # number all the labels
    numbered_labels = []
    for label in df[LABEL_COLUMN]:
        numbered_labels.append(labels.index(label))

    t,x,y,z,s = fe.parseData_3DLabeled(df.index, df['aX'], df['aY'], df['aZ'], numbered_labels, fs, verbose=verbose)

    df = pd.DataFrame(index=[long(i) for i in t])#pd.to_datetime(t, unit='ms')
    df['aX'] = x
    df['aY'] = y
    df['aZ'] = z
    df['aA'] = np.hypot(x, np.hypot(y, z))

    if not named:
        df[LABEL_COLUMN] = s
        return df

    # else
    named_labels = []
    for label in s:
        named_labels.append(labels[long(label)])

    df[LABEL_COLUMN] = named_labels
    return df

def buildFeatureFrame(df, Fs, N):
    idx = [(df.index)[0]]

    if(len(df) < int(N * Fs)):
        return pd.DataFrame(index=idx)

    label = fe.getMajorityLabel_Nominal(df['class'])

    if label == '?':
        return pd.DataFrame(index=idx)

    t = df.index
    x = df['aX']
    y = df['aY']
    z = df['aZ']
    a = df['aA']

    LX, LY, LZ, LA, LS = angles.linearize(x=x, y=y, z=z, a=a, cutoff=1.0, fs=Fs, order=1)
    GX, GY, GZ, GA, GS = angles.gravity(x=x, y=y, z=z, a=a, cutoff=0.25, fs=Fs, order=1)

    PTCH, ROLL = angles.pitch_roll(x, y, z)
    DELT = angles.get_angle_changes(x.tolist(), y.tolist(), z.tolist(), a.tolist())

    GPTCH, GROLL = angles.pitch_roll(GX, GY, GZ)
    GDELT = angles.get_angle_changes(GX, GY, GZ, GS)

    GXL, GXF = fe.getLessFeatures(name="GX", signal=GX, samplingFreq=Fs)
    GYL, GYF = fe.getLessFeatures(name="GY", signal=GY, samplingFreq=Fs)
    GZL, GZF = fe.getLessFeatures(name="GZ", signal=GZ, samplingFreq=Fs)
    GAL, GAF = fe.getLessFeatures(name="GA", signal=GA, samplingFreq=Fs)
    GSL, GSF = fe.getLessFeatures(name="GS", signal=GS, samplingFreq=Fs)

    LXL, LXF = fe.getLessFeatures(name="LX", signal=LX, samplingFreq=Fs)
    LYL, LYF = fe.getLessFeatures(name="LY", signal=LY, samplingFreq=Fs)
    LZL, LZF = fe.getLessFeatures(name="LZ", signal=LZ, samplingFreq=Fs)
    LAL, LAF = fe.getLessFeatures(name="LA", signal=LA, samplingFreq=Fs)
    LSL, LSF = fe.getLessFeatures(name="LS", signal=LS, samplingFreq=Fs)

    GPL, GPF = fe.getLessFeatures(name="G_PTCH", signal=GPTCH, samplingFreq=Fs)
    GRL, GRF = fe.getLessFeatures(name="G_ROLL", signal=GROLL, samplingFreq=Fs)
    GDL, GDF = fe.getLessFeatures(name="G_DELT", signal=GDELT, samplingFreq=Fs)

    LPL, LPF = fe.getLessFeatures(name="PTCH", signal=PTCH, samplingFreq=Fs)
    LRL, LRF = fe.getLessFeatures(name="ROLL", signal=ROLL, samplingFreq=Fs)
    LDL, LDF = fe.getLessFeatures(name="DELT", signal=DELT, samplingFreq=Fs)

    data = pd.DataFrame([GXF + GYF + GZF + GAF + GSF + LXF + LYF + LZF + LAF + LSF + GPF + GRF + GDF + LPF + LRF + LDF + [label]], columns=(GXL + GYL + GZL + GAL + GSL + LXL + LYL + LZL + LAL + LSL + GPL + GRL + GDL + LPL + LRL + LDL + ['class']), index=idx)
    return data

def generateFeatures(raw_dframe, window, fs, overlap, ignore_idx=True):
    dataLength = len(raw_dframe)
    sampleCount = int(window * fs)
    data = pd.concat(Parallel(n_jobs=num_cores)(delayed(buildFeatureFrame)(raw_dframe.iloc[x:x+sampleCount], Fs=fs, N=window) for x in xrange(0, dataLength, int(overlap*sampleCount))), ignore_index=ignore_idx)
    return data.dropna(how='all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',  dest='src', help="accelerometer CSV file", required=True)
    parser.add_argument('-o','--output', dest='out', help = "output features file", required=True)
    parser.add_argument('-O','--overlap', dest='overlap', help = "sliding window overlap", required=True)
    parser.add_argument('-f','--samplingrate', dest='fs', help = "interpolation sampling rate", required=True)
    parser.add_argument('-w','--windowlen', dest='windowlen', help = "window length in seconds", required=True)
    args, leftovers = parser.parse_known_args()

    # Import CSVs into list of dataframes.
    # We assume that each CSV contains a continous stream of data, and the
    # separate CSVs are disjoint
    files = importCSV(args.src, recursive=True)
    raw_df = (Parallel(n_jobs=1)(delayed(importLabeledFileInterpolated)(file_, fs=float(args.fs)) for file_ in files))
    print 'Imported', len(raw_df), 'input files'

    generated_features = pd.concat((generateFeatures)(df, window=float(args.windowlen), fs=float(args.fs), overlap=float(args.overlap), ignore_idx=True) for df in raw_df)
    print 'Exporting', len(generated_features), 'instances of feature data'

    fe.writeARFF(df=generated_features.sample(frac=1), name=args.out)
    print 'done...'
