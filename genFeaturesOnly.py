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

ACCL = ["aX_t", "aY_t", "aZ_t", 'aA_t']
GYRO = ["gX_t", "gY_t", "gZ_t", 'gA_t']
SIGNALS = ACCL+GYRO

GRAVITY = 9.79607

CSV_COL_NAMES = [INDEX_COLUMN] + ACCL[0:3] + GYRO[0:3] + [LABEL_COLUMN]


LABELS = {
                "label"             : LABEL_COLUMN,
                "axis"              : 7,

                "names"             : ["movement", "position", "stairs", "stairDirection", "assist", "assistType", "assistType2"],
                # Walking
                "walking"           : ["moving", "walking",   "flat",     "flat", "noAssist", "noAssist",     "noAssist"],
                "flat"              : ["moving", "walking",   "flat",     "flat", "noAssist", "noAssist",     "noAssist"],
                "stairs"            : ["moving", "walking", "stairs",        "?", "noAssist", "noAssist",     "noAssist"],
                "upstairs"          : ["moving", "walking", "stairs",       "up", "noAssist", "noAssist",     "noAssist"],
                "downstairs"        : ["moving", "walking", "stairs",     "down", "noAssist", "noAssist",     "noAssist"],
                # Assistive walking devices
                "walking-noassist"  : ["moving", "walking",   "flat",     "flat", "noAssist", "noAssist",     "noAssist"],
                "assisted"          : ["moving", "walking",   "flat",     "flat", "assisted",        "?",            "?"],
                "hemi"              : ["moving", "walking",   "flat",     "flat", "assisted",     "hemi"],
                "single-leg-walker" : ["moving", "walking",   "flat",     "flat", "assisted",   "walker",         "cane"],
                "four-leg-walker"   : ["moving", "walking",   "flat",     "flat", "assisted",   "walker",     "quadCane"],
                "crutches"          : ["moving", "walking",   "flat",     "flat", "assisted",   "walker",     "crutches"],
                "walker"            : ["moving", "walking",   "flat",     "flat", "assisted",   "walker", "frontWheeled"],

                "laying_down_moving": ["moving",  "lyingDown"],
                "laying_down_still" : ["still" ,  "lyingDown"],
                "laying_down"       : ["?"     ,  "lyingDown"],

                "wheelchair"        : ["moving", "sitting", "sitting", "sitting", "assisted", "wheelchair", "wheelchair"],
                "sitting_moving"    : ["moving", "sitting"],
                "sitting_still"     : ["still",  "sitting"],
                "sitting"           : ["?",      "sitting"],

                "standing_moving": ["moving", "standing"],
                "standing_still" : ["still",  "standing"],
                "standing"       : ["?"    ,  "standing"],
            }

def importCSV(srcDir, recursive):
    if(recursive == False):return glob.glob(srcDir+'/*.csv')
    matches = []
    for root, dirnames, filenames in os.walk(srcDir):
        for filename in fnmatch.filter(filenames, '*.csv'):
            matches.append(os.path.join(root, filename))
    return matches

def importLabeledFile(file_):
    df = pd.read_csv(file_, header=None, names=CSV_COL_NAMES, index_col=INDEX_COLUMN)
    df.sort_index(inplace=True)

    # accel related
    #df[ACCL[0:3]] = df[ACCL[0:3]].abs()                                                  # sets x,y,z values to absolute
    df = df[(df[ACCL[0]] < A_THRES) & (df[ACCL[1]] < A_THRES) & (df[ACCL[2]] < A_THRES)] # removes all values greater than the threshold
    df[ACCL[3]] = df[ACCL[0:3]].apply(lambda row:(fe.getMag(row)), axis=1)       # calculates the linear acceleration (minus gravity)

    # rawInputFile is 3D, rename label column to 'gX_t' and vice-versa
    if(df[GYRO[1]].isnull().values.any()):
        print 'No Gyroscope data! :', file_
        df.rename(columns={GYRO[0]: LABEL_COLUMN, LABEL_COLUMN: GYRO[0]}, inplace=True)
        df[GYRO[3]] = df[GYRO[2]]
        return df

    # rawInputFile is unlabeled, set class values to '?'
    if(df[LABEL_COLUMN].isnull().values.any()):
        df[LABEL_COLUMN] = '?'
    # gyro related
    df[GYRO[0:3]] = df[GYRO[0:3]].abs()                                                  # sets x,y,z values to absolute
    df = df[(df[GYRO[0]] < G_THRES) & (df[GYRO[1]] < G_THRES) & (df[GYRO[2]] < G_THRES)] # removes all values greater than the threshold
    df[GYRO[3]] = df[GYRO[0:3]].apply(lambda row:fe.getMag(row), axis=1)                 # calculates the magnitude of velocity
    return df

def importLabeledFileInterpolated(file_, fs):
    # import the file
    df = importLabeledFile(file_)

    # add all labels, including unknown '?'
    labels = list(set(df[LABEL_COLUMN]))
    labels.insert(0, '?')
    print labels
    # number all the labels
    numbered_labels = []
    for label in df[LABEL_COLUMN]:
        numbered_labels.append(labels.index(label))


    t,x,y,z,s = fe.parseData_3DLabeled(df.index, df['aX_t'], df['aY_t'], df['aZ_t'], numbered_labels, fs)

    print len(s), len(t)
    named_labels = []
    for label in s:
        named_labels.append(labels[long(label)])

    df = pd.DataFrame(index=[long(i) for i in t])#pd.to_datetime(t, unit='ms')
    df['aX_t'] = x
    df['aY_t'] = y
    df['aZ_t'] = z
    df[LABEL_COLUMN] = named_labels
    return df

# def generateFeatures(raw_dframe, overlap=0.00625, ignore_idx=True):
#     dataLength = len(raw_dframe)
#     data = pd.concat(Parallel(n_jobs=num_cores)(delayed(fe.getFeatures_pandas)(raw_dframe.iloc[x:x+sampleCount], Fs=samplingFreq, N=windowSize) for x in xrange(0, dataLength, int(overlap*sampleCount))), ignore_index=ignore_idx)
#     return data.dropna(how='all')

def generateFeatures(raw_dframe, window, fs, overlap, ignore_idx=True):
    dataLength = len(raw_dframe)
    sampleCount = int(window * fs)
    data = pd.concat(Parallel(n_jobs=num_cores)(delayed(angles.getLessFeatures_pandas)(raw_dframe.iloc[x:x+sampleCount], Fs=fs, N=window) for x in xrange(0, dataLength, int(overlap*sampleCount))), ignore_index=ignore_idx)
    return data.dropna(how='all')

def writeToCSV(df, filename):
    # move class column to the end
    cols = df.columns.tolist()
    class_idx = cols.index(LABEL_COLUMN)
    cols.append(cols.pop(class_idx))

    # write the rawInputFile
    df[cols].fillna('?').to_csv(filename, encoding='utf-8', index=False)
    print 'Exported to CSV :', filename
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',  dest='src', help="accelerometer CSV file", required=True)
    parser.add_argument('-o','--output', dest='out', help = "output features file", required=True)
    parser.add_argument('-O','--overlap', dest='overlap', help = "sliding window overlap", required=True)
    parser.add_argument('-f','--samplingrate', dest='fs', help = "interpolation sampling rate", required=True)
    parser.add_argument('-w','--windowlen', dest='windowlen', help = "window length in seconds", required=True)
    args, leftovers = parser.parse_known_args()
    #raw_df = importLabeledFile(args.src)

    files = importCSV(args.src, recursive=True)
    raw_df = (Parallel(n_jobs=num_cores)(delayed(importLabeledFileInterpolated)(file_, fs=float(args.fs)) for file_ in files))
    print 'Imported', len(raw_df), 'input files'

    generated_features = pd.concat((generateFeatures)(df, window=float(args.windowlen), fs=float(args.fs), overlap=float(args.overlap), ignore_idx=True) for df in raw_df)
    print 'Exporting', len(generated_features), 'instances of feature data'
    #writeToCSV(generated_features, args.out)

    fe.writeARFF(df=generated_features, name=args.out)
    #pd2arff.pandas2arff(generated_features,args.out+".arff")
    print 'done...'
