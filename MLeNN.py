# -*- coding: utf-8 -*-
"""
@inproceedings{charte2014MLeNN,
  title={MLeNN: a first approach to heuristic multilabel undersampling},
  author={Charte, Francisco and Rivera, Antonio J and del Jesus, Mar{\'\i}a J and Herrera, Francisco},
  booktitle={International Conference on Intelligent Data Engineering and Automated Learning},
  pages={1--9},
  year={2014},
  organization={Springer}
}
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def caculate_IRLbl(Y):
    # imbalance ratio per label
    posNumsPerLabel = np.sum(Y, axis=0)
    maxPosNums = np.max(posNumsPerLabel)
    IRLbl = maxPosNums / posNumsPerLabel
    return IRLbl


def caculate_meanIR(Y):
    # average imbalance ratio
    IRLbl = caculate_IRLbl(Y)
    meanIR = np.mean(IRLbl)
    return meanIR


def get_minBag(Y):
    n, m = Y.shape
    IRLbl = caculate_IRLbl(Y)
    meanIR = caculate_meanIR(Y)
    minBag = []
    for i in range(m):
        if IRLbl[i] > meanIR:
            minBag.append(i)
    return minBag


def get_minMajInstInd(Y, minBag):
    n, m = Y.shape
    minInstInd = []
    majInstInd = []
    for i in range(n):
        if (Y[i, minBag]==1).any():
            minInstInd.append(i)
        else:
            majInstInd.append(i)
    return minInstInd, majInstInd


def adjust_hamming_distance(y1, y2):
    flag1 = np.sum(y1)
    flag2 = np.sum(y2)
    flag = (flag1 and flag2)
    if flag:
        ele = np.sum((y1 + y2)==1)
        den = flag1 + flag2
        return ele / den
    else:
        return 1


def NN_index(X, k=5):
    # n_neighbors including the sample itself, 
    # so we take the number of n_neighbors as k+1 (as the following shows),
    # then delete itself from the neighbors.
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean', 
                           algorithm='auto').fit(X)
    euclidean, index = nn.kneighbors(X)
    return index[:,1:]


def MLeNN(X, Y, NN=3, HT=0.75):
    # MLeNN (MultiLabel edited Nearest Neighbor)
    nnIndex = NN_index(X, NN)
    minBag = get_minBag(Y)
    minInstInd, majInstInd = get_minMajInstInd(Y, minBag)
    markForRemoving = []
    for sampleIndex in majInstInd:
        numDifferences = 0
        sampleNNIndexs = nnIndex[sampleIndex,:]
        for sampleNNIndex in sampleNNIndexs:
            adjustedHammingDist = adjust_hamming_distance(Y[sampleIndex,:],
                                                          Y[sampleNNIndex,:])
            if adjustedHammingDist > HT:
                numDifferences = numDifferences + 1
        if numDifferences >= (NN/2):
            print("Remove:", sampleIndex)
            markForRemoving.append(sampleIndex)
    
    X_new = np.delete(X, markForRemoving, axis=0)
    Y_new = np.delete(Y, markForRemoving, axis=0)
    return X_new, Y_new

if __name__ == '__main__':
    np.random.seed(666)
    HT = 0.75
    NN = 3
    p = 103
    path = r'C:\Users\dell\Desktop\datasets\Yeast\Yeast.csv'
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    X, Y = data[:,:p], data[:,p:]
    X_new, Y_new = MLeNN(X, Y, NN=3, HT=0.75)
    