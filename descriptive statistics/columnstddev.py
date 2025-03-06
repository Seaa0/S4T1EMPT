import numpy as np
import pandas as pd

dataset = pd.read_csv("datasets/preprocessedDataset.csv")


for colname in dataset:
    col = dataset[colname]
    range = np.ptp(dataset[colname])
    if range < 100000: # quantitative data
        continue
    print(colname+': ')
    i = 0
    durastddev = []
    qualstddev = []
    while True:
        dura = []
        qual = []
        for itm, itm2, itm3 in zip(col,dataset['sleep duration'],dataset['quality of sleep']):
            if itm == i:
                dura.append(itm2)
                qual.append(itm3)
        if len(dura) > 1:
            durastddev.append(pd.Series(dura).std())
        if len(qual) > 1:
            qualstddev.append(pd.Series(qual).std())

        i += 100000
        if i > max(col):
            break
    print('Duration of sleep:',sum(durastddev)/len(durastddev))
    print('Quality of sleep:',sum(qualstddev)/len(qualstddev))
    print()