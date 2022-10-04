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

def label_detache(X, Y):
    n, m = Y.shape
    return {k: np.where(Y[:,k]==1)[0] for k in range(m)}


def minority_label(labelSetBag, meanSize):
    m = len(labelSetBag)
    minInd = []
    for i in range(m):
        num = len(labelSetBag[i])
        if num < meanSize:
            minInd.append(i)
    return minInd

def get_meanInc(labelSetBag, minInd, samplesGenerate):
    meanInc = 0
    k = len(minInd)
    num = [len(labelSetBag[minInd[i]]) for i in range(k)]
    meanInc = samplesGenerate / len(num)
    ind = np.argsort(num)
    sortInd = [minInd[ind[k-j-1]] for j in range(k)]
    return meanInc, sortInd

def generate_ind(n, labelSetBag, meanSize, samplesGenerate):
    minInd = minority_label(labelSetBag, meanSize)
    meanInc, sortInd = get_meanInc(labelSetBag, minInd, samplesGenerate)
    k = len(minInd)
    rBag = []
    generateInd = []
    for i in range(k):
        nMinBagi = len(labelSetBag[sortInd[i]])
        rBag.append(min(meanSize - nMinBagi, meanInc))
        # 以下两行在算法中虽有提及，但不知道有什么用，简直莫名其妙。。。。。。
        # remainder = meanRed - rBag[i]
        # distributeAmongBagsj>i(remainder)
        rBagInd = list(labelSetBag[sortInd[i]])
        print(rBag[i])
        for _ in range(int(rBag[i])):
            x = np.random.randint(0, nMinBagi)
            generateInd.append(rBagInd[x])
    return generateInd


def LPROS(X, Y, ratio=0.10):
    # ratio is the parameter P
    n, m = Y.shape
    meanSize = np.sum(Y)/m
    samplesGenerate = n*ratio
    labelSetBag = label_detache(X, Y)
    minInd = minority_label(labelSetBag, meanSize)
    meanInc, sortInd = get_meanInc(labelSetBag, minInd, samplesGenerate)
    generateInd = generate_ind(n, labelSetBag, meanSize, samplesGenerate)
    X_gen, Y_gen = X[generateInd,:], Y[generateInd,:]
    X_new, Y_new = np.r_[X, X_gen], np.r_[Y, Y_gen]
    
    return X_new, Y_new
