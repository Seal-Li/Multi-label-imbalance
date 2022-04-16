# -*- coding: utf-8 -*-
"""
@article{pereira2020MLTL,
  title={MLTL: A multi-label approach for the Tomek Link undersampling algorithm},
  author={Pereira, Rodolfo M and Costa, Yandre MG and Silla Jr, Carlos N},
  journal={Neurocomputing},
  volume={383},
  pages={95--105},
  year={2020},
  publisher={Elsevier}
}
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def label_measures(Y):
    n, m = Y.shape
    LCard = np.sum(Y) / n
    LDen = LCard / m
    return LCard, LDen


def caculate_IRLbl(Y):
    # imbalance ratio per label
    posNumsPerLabel = np.sum(Y, axis=0)
    maxPosNums = np.max(posNumsPerLabel)
    return maxPosNums / posNumsPerLabel


def caculate_meanIR(Y):
    # average imbalance ratio
    IRLbl = caculate_IRLbl(Y)
    return np.mean(IRLbl)


def determine_TH(Y):
    meanIR = caculate_meanIR(Y)
    I = 1 / np.sqrt(meanIR)
    TH = 0
    if I < 0.3:
        return 0.15
    elif I >= 0.5:
        return 0.5
    else:
        return 0.3


def get_minBag(Y):
    n, m = Y.shape
    IRLbl = caculate_IRLbl(Y)
    meanIR = caculate_meanIR(Y)
    return [i for i in range(m) if IRLbl[i] > meanIR]


def get_MajInstInd(Y, minBag):
    n, m = Y.shape
    return [i for i in range(n) if not (Y[i, minBag]==1).any()]


def NN_index(X, k=5):
    # n_neighbors including the sample itself, 
    # so we take the number of n_neighbors as k+1 (as the following shows),
    # then delete itself from the neighbors.
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean', 
                           algorithm='auto').fit(X)
    euclidean, index = nn.kneighbors(X)
    return index[:,1:]


def adjust_hamming_distance(y1, y2):
    flag1 = np.sum(y1)
    flag2 = np.sum(y2)
    if flag := (flag1 and flag2):
        ele = np.sum((y1 + y2)==1)
        den = flag1 + flag2
        return ele / den
    else:
        return 1


def undersampling_method(X, Y, TH):
    minBag = get_minBag(Y)
    TomekLinkInd = set()
    NNList = NN_index(X, k=5)
    majInstInds = get_MajInstInd(Y, minBag)
    for majInstInd in majInstInds:
        NN = NNList[majInstInd,0]
        dist = adjust_hamming_distance(Y[majInstInd,:], Y[NN,:])
        if (dist>=TH):
            TomekLinkInd.add(majInstInd)
    return TomekLinkInd


def cleaning_method(X, Y, TH):
    n, m = Y.shape
    TomekLinkInd = set()
    NNList = NN_index(X, k=5)
    for i in range(n):
        NN = NNList[i,0]
        dist = adjust_hamming_distance(Y[i,:], Y[NN,:])
        if (dist>=TH):
            TomekLinkInd.add(i)
    return TomekLinkInd


def MLTL(X, Y, method="under-sampling"):
    """
    "method" can be choosed as {1:"cleaning",2:"under-sampling"},
    default as "under-sampling".
    """
    n = Y.shape[0]
    allInd = set(np.arange(n))
    TH = determine_TH(Y)
    print("The threshold is:", TH)
    if (method=="cleaning"):
        print("Using cleaning method!")
        TomekLinkInd = cleaning_method(X, Y, TH)
    if (method=="under-sampling"):
        print("Using undersampling method!")
        TomekLinkInd = undersampling_method(X, Y, TH)
    
    PreservedInd = list(allInd.difference(TomekLinkInd))
    X_new, Y_new = X[PreservedInd,:], Y[PreservedInd,:]
    print("Done!\n")
    return X_new, Y_new


if __name__ == '__main__':
    np.random.seed(666)
    # p = 293
    p = 103
    path = r'C:\Users\dell\Desktop\datasets\Yeast\Yeast.csv'
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    X, Y = data[:,:p], data[:,p:]
    
    X_new1, Y_new1 = MLTL(X, Y, method="cleaning")
    X_new2, Y_new2 = MLTL(X, Y, method="under-sampling")
