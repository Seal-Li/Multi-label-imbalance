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
    labelSetBag = dict()
    for k in range(m):
        labelSetBag[k] = np.where(Y[:,k]==1)[0]
    return labelSetBag


def majority_label(labelSetBag, meanSize):
    m = len(labelSetBag)
    majInd = []
    for i in range(m):
        num = len(labelSetBag[i])
        if num > meanSize:
            majInd.append(i)
    return majInd


def get_meanRed(labelSetBag, majInd, sampleToDelete):
    # Red is the abbreviation of Reduction
    k = len(majInd)
    num = []
    meanRed = 0
    for i in range(k):
        num.append(len(labelSetBag[majInd[i]]))
    meanRed = sampleToDelete / len(num)
    ind = np.argsort(num)
    sortInd = []
    for j in range(k):
        sortInd.append(majInd[ind[j]])
    return meanRed, sortInd


def delete_ind(n, labelSetBag, meanSize, sampleToDelete):
    majInd = majority_label(labelSetBag, meanSize)
    meanRed, sortInd = get_meanRed(labelSetBag, majInd, sampleToDelete)
    k = len(majInd)
    rBag = []
    deleteSet = set()
    allInd = set(np.arange(n))
    for i in range(k):
        nMajBagi = len(labelSetBag[sortInd[i]])
        rBag.append(min(nMajBagi - meanSize, meanRed))
        # 以下两行在算法中虽有提及，但不知道有什么用，简直莫名其妙。。。。。。
        # remainder = meanRed - rBag[i]
        # distributeAmongBagsj>i(remainder)
        # rBagInd = list(np.arange(nMajBagi))
        rBagInd = list(labelSetBag[sortInd[i]])
        for j in range(int(rBag[i])):
            x = np.random.randint(0, nMajBagi)
            while x in deleteSet:
                x = np.random.randint(0, nMajBagi)
            deleteSet.add(rBagInd[x])
    remainderInd = list(allInd.difference(deleteSet))
    return remainderInd


def LPRUS(X, Y, ratio=0.2):
    # ratio is the parameter P
    n, m = Y.shape
    meanSize = np.sum(Y)/m
    sampleToDelete = n*ratio
    
    labelSetBag = label_detache(X, Y)
    majInd = majority_label(labelSetBag, meanSize)
    meanRed, sortInd = get_meanRed(labelSetBag, majInd, sampleToDelete)
    remainderInd = delete_ind(n, labelSetBag, meanSize, sampleToDelete)
    X_new, Y_new = X[remainderInd,:], Y[remainderInd,:]
    return X_new, Y_new


if __name__ == '__main__':
    p = 103
    path = r'C:\Users\dell\Desktop\datasets\Yeast\Yeast.csv'
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    X, Y = data[:,:p], data[:,p:]
    X_new, Y_new = LPRUS(X, Y, ratio=0.2)