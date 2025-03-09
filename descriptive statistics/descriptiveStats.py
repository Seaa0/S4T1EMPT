import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    IQR = col.quantile(0.75) - col.quantile(0.25)
    print('Interquartile Range:',IQR)
    print()

    plt.figure(figsize=(6, 4))
    sns.boxplot(y=col)
    
    # Add labels and title
    plt.title(f'Boxplot of {colname}')
    plt.ylabel(colname)
    
    # Show the plot
    plt.show()

