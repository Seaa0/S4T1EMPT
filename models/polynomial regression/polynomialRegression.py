import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("preprocessedDataset.csv")

# just trying with 1 feature for now
# linear regression
weights = [0]*3

y = dataset['sleep duration']
X = dataset['physical activity level']

i = 0
learningRate = 0.00000002

prev = 0
while True:
    i += 1
    SE = []
    djdwj = []
    for j in range(3):
        currdjdwj = []
        for yi, xi in zip(y,X):
            yj = weights[2]*xi**2+weights[1]*xi+weights[0]
            try:
                SE.append((yi-yj)**2)

                currdjdwj.append((yj-yi)*(xi**j))
                
            except OverflowError:
                print(yi,yj)
                print('Epoch',i)
                print('Learning rate too large, crashing...')
                exit()
        djdwj.append(sum(currdjdwj)/len(currdjdwj)*2)

    MSE = sum(SE)/len(SE)
    Mdjdwj = []
    assert len(djdwj) == 3
    for j in range(3):
        weights[j] -= djdwj[j]*learningRate
    
    if i % 10000 == 0:
        print('Epoch {}: MSE = {}'.format(i, MSE))
    if prev == MSE:
        print('Epoch {}: MSE = {}'.format(i, MSE))
        print('Ending loop...')
        break
    prev = MSE


