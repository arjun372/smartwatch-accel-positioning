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

    aX_L = [str('aX_' + str(i)) for i in xrange(0, len(x))]
    aY_L = [str('aY_'+str(i)) for i in xrange(0, len(y))]
    aZ_L = [str('aZ_'+str(i)) for i in xrange(0, len(z))]

    data = pd.DataFrame([x.tolist() + y.tolist() + z.tolist() + [label]], columns=(aX_L + aY_L + aZ_L + ['class']), index=idx)

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

    fe.writeARFF(df=generated_features, name=args.out)
    print 'done...'
