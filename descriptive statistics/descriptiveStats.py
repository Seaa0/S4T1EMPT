import numpy as np
import pandas as pd

dataset = pd.read_csv("datasets/preprocessedDataset.csv")


for colname in dataset:
    col = dataset[colname]
    ranges = np.ptp(dataset[colname])
    if ranges >= 100000: # qualitative data
        continue
    print('----Column:',colname+'----')
    mean = col.mean()
    print('Mean:',mean)
    median = col.median()
    print('Median:',median)
    mode = list(col.mode())
    print('Mode(s):',', '.join(str(x) for x in mode))
    stddev = col.std()
    print('Standard Deviation:',stddev)
    variance = stddev**2
    print('Variance:',variance)
    print('Range:',ranges)
    IQR = list(col.quantile([0.75]))[0]-list(col.quantile([0.25]))[0]
    print('Interquartile Range:',IQR)
    print()