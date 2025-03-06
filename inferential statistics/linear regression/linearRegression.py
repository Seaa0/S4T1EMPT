import numpy as np
import pandas as pd

dataset = pd.read_csv("datasets/preprocessedDataset.csv")

finalStats = {}

print('Training...')
for colname in dataset:
    for ylabel in ['sleep duration','quality of sleep']:
        check = 0
        m = 0
        b = 0
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
        learningRate = 0.0005
        prev = 0
        AE = []
        sstot = []
        while True:
            i += 1
            SE = []
            dmsedm = []
            dmsedb = []
            for yi, xi in zip(y,X):
                yj = m*xi+b
                try:
                    SE.append((yi-yj)**2)
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
            
                dmsedm.append((yj-yi)*xi)
                dmsedb.append((yj-yi))
            MSE = sum(SE)/len(SE)
            Mdmsedm = sum(dmsedm)/len(dmsedm)*2
            Mdmsedb = sum(dmsedb)/len(dmsedb)*2
            m -= learningRate*Mdmsedm
            b -= learningRate*Mdmsedb
            if i % 10000 == 0:
                print('Epoch {}: MSE = {}'.format(i, MSE))
            if check:
                sumsstot = sum(sstot)
                Rsquared = 1-(MSE*len(SE)/sumsstot)
                RMSE = MSE**0.5
                MAE = sum(AE)/len(AE)
                print('Epoch {}:'.format(i))
                print(f'MSE = {MSE}')
                print(f'RMSE = {RMSE}')
                print(f'R^2 = {Rsquared}')
                print(f'MAE = {MAE}')
                print('Moving on...')
                finalStats[(colname,ylabel)] = (MSE,RMSE,Rsquared,MAE)
                break
            if prev == MSE or i >= 100000: # no point spending so long optimising the MSE by 0.00001
                if MSE > 5:
                    print('Epoch',i)
                    print('Learning rate too large, trying again...')
                    i = 0
                    learningRate /= 10
                    m = 0
                    b = 0
                    continue
                check = 1
            prev = MSE

        print()

for itm in finalStats:
    print('Feature:',itm[0])
    print('Target:',itm[1])
    print(f'MSE = {finalStats[itm][0]}')
    print(f'RMSE = {finalStats[itm][1]}')
    print(f'R^2 = {finalStats[itm][2]}')
    print(f'MAE = {finalStats[itm][3]}')
    print()
    
