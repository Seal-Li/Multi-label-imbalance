# -*- coding: utf-8 -*-
"""
@article{charte2015addressing,
  title={Addressing imbalance in multilabel classification: Measures and random resampling algorithms},
  author={Charte, Francisco and Rivera, Antonio J and del Jesus, Mar{\'\i}a J and Herrera, Francisco},
  journal={Neurocomputing},
  volume={163},
  pages={3--16},
  year={2015},
  publisher={Elsevier}
}
"""
import numpy as np

def caculate_IRLbl(Y):
    # imbalance ratio per label
    posNumsPerLabel = np.sum(Y, axis=0)
    maxPosNums = np.max(posNumsPerLabel)
    return maxPosNums / posNumsPerLabel


def caculate_meanIR(Y):
    # average imbalance ratio
    IRLbl = caculate_IRLbl(Y)
    return np.mean(IRLbl)


def get_minBag(Y):
    n, m = Y.shape
    IRLbl = caculate_IRLbl(Y)
    meanIR = caculate_meanIR(Y)
    return [i for i in range(m) if IRLbl[i] > meanIR]


def MLROS(X, Y, ratio=0.2):
    # ratio is the parameter P
    N = X.shape[0]
    samplesToClone = N*ratio
    meanIR = caculate_meanIR(Y)
    minBag = get_minBag(Y)
    X_new, Y_new = X, Y
    k = len(minBag)
    skipInd = set()
    while (samplesToClone > 0) and (len(skipInd)!=k) :
        for i in range(k):
            if i in skipInd:
                continue
            
            minBagInd = np.where(Y[:,minBag[i]]==1)[0]
            nMinBag = len(minBagInd)
            IRLbl_new = caculate_IRLbl(Y_new)
            
            if IRLbl_new[minBag[i]] <= meanIR:
                skipInd.add(i)
        
            x = np.random.randint(0, nMinBag, 1)
            instance = X[minBagInd[x],:].reshape(1,-1)
            target = Y[minBagInd[x],:].reshape(1,-1)
            X_new, Y_new = np.r_[X_new, instance], np.r_[Y_new, target]
            
            samplesToClone = samplesToClone - 1
            if (samplesToClone <= 0):
                break
    return X_new, Y_new


if __name__ == '__main__':
    p = 103
    path = r'C:\Users\dell\Desktop\datasets\Yeast\Yeast.csv'
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    X, Y = data[:,:p], data[:,p:]
    X_new, Y_new = MLROS(X, Y, ratio=0.2)

