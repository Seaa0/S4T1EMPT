import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("datasets/preprocessedDataset.csv")

finalStats = {}

for colname in dataset:
    for ylabel in ['sleep duration','quality of sleep']:
        check = 0
        weights = [0]*3
        col = dataset[colname]
        range = np.ptp(dataset[colname])
        if range >= 100000: # qualitative data
            continue
        if colname in ['sleep duration','quality of sleep']:
            continue
        X = dataset[colname]
        y = dataset[ylabel]
        print('X:',colname+',','y:',ylabel)

        i = 0
        learningRate = 0.000005
        prev = 0
        AE = []
        sstot = []

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
                        if check:
                            AE.append(abs(yi-yj))
                            sstot.append((yi-y.mean())**2)
                        
                    except OverflowError:
                        print('Epoch',i)
                        print('Learning rate too large, trying again...')
                        i = 0
                        learningRate /= 10
                        m = 0
                        b = 0
                        continue
                djdwj.append(sum(currdjdwj)/len(currdjdwj)*2)

            MSE = sum(SE)/len(SE)/3
            Mdjdwj = []
            assert len(djdwj) == 3
            for j in range(3):
                weights[j] -= djdwj[j]*learningRate
            
            if i % 10000 == 0:
                print('Epoch {}: MSE = {}'.format(i, MSE))
            if check:
                sumsstot = sum(sstot)
                Rsquared = 1-(MSE*len(SE)/sumsstot)
                RMSE = MSE**0.5
                MAE = sum(AE)/len(AE)/3
                print('Epoch {}:'.format(i))
                print(f'MSE = {MSE}')
                print(f'RMSE = {RMSE}')
                print(f'R^2 = {Rsquared}')
                print(f'MAE = {MAE}')
                print('Moving on...')
                finalStats[(colname,ylabel)] = (MSE,RMSE,Rsquared,MAE)
            if prev == MSE or i >= 100000: # no point spending so long optimising the MSE by 0.00001
                if MSE > 5:
                    print('Epoch',i)
                    print('Learning rate too large, trying again...')
                    learningRate /= 10
                    m = 0
                    b = 0
                    continue
                check = 1
                print('Epoch {}: MSE = {}'.format(i, MSE))
                print('Ending loop...')
                break
            prev = MSE

for itm in finalStats:
    print('Feature:',itm[0])
    print('Target:',itm[1])
    print(f'MSE = {finalStats[itm][0]}')
    print(f'RMSE = {finalStats[itm][1]}')
    print(f'R^2 = {finalStats[itm][2]}')
    print(f'MAE = {finalStats[itm][3]}')
    print()

