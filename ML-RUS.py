# -*- coding: utf-8 -*-
"""
@article{charte2015LP-ROS/LP-RUS/ML-ROS/ML/RUS,
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


def label_detache(Y):
    n, m = Y.shape
    return {k: np.where(Y[:,k]==1)[0] for k in range(m)}


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

def MLRUS(Y, ratio):
    n, m = Y.shape
    sampleToDelete = n*ratio
    minBag = get_minBag(Y)
    minInstInd, majInstInd = get_minMajInstInd(Y, minBag)
    deleteList = []
    temp_Y = Y
    while sampleToDelete>0:
        ind = int(np.random.choice(majInstInd, 1))
        while (ind in deleteList):
            ind = int(np.random.choice(majInstInd, 1))
        deleteList.append(ind)
        sampleToDelete = sampleToDelete - 1
        temp_Y = np.delete(Y, deleteList, axis=0)
        temp_minBag = get_minBag(temp_Y)
        if (set(temp_minBag)!=set(minBag)):
            minInstInd, majInstInd = get_minMajInstInd(Y, temp_minBag)
        print("The number of sample to delete:", sampleToDelete)
    X_new = np.delete(X, deleteList, axis=0)
    Y_new = np.delete(Y, deleteList, axis=0)
    return X_new, Y_new

if __name__ == '__main__':
    p = 103
    path = r'C:\Users\dell\Desktop\datasets\Yeast\Yeast.csv'
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    X, Y = data[:,:p], data[:,p:]
    X_new, Y_new = MLRUS(Y, ratio=0.2)
