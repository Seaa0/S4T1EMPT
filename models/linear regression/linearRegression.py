import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("datasets/preprocessedDataset.csv")

# just trying with 1 feature for now
# linear regression
m = 0
b = 0

y = dataset['sleep duration']
X = dataset['physical activity level']

i = 0
learningRate = 0.00025

prev = 0
while True:
    i += 1
    SE = []
    dmsedm = []
    dmsedb = []
    for yi, xi in zip(y,X):
        yj = m*xi+b
        try:
            SE.append((yi-yj)**2)
        except OverflowError:
            print('Epoch',i)
            print('Learning rate too large, crashing...')
            exit()
    
        dmsedm.append((yj-yi)*xi)
        dmsedb.append((yj-yi))
    MSE = sum(SE)/len(SE)
    Mdmsedm = sum(dmsedm)/len(dmsedm)*2
    Mdmsedb = sum(dmsedb)/len(dmsedb)*2
    m -= learningRate*Mdmsedm
    b -= learningRate*Mdmsedb
    if i % 10000 == 0:
        print('Epoch {}: MSE = {}'.format(i, MSE))
    if prev == MSE:
        print('Epoch {}: MSE = {}'.format(i, MSE))
        print('Ending loop...')
        break
    prev = MSE


