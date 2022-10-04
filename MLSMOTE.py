# -*- coding: utf-8 -*-
"""
@article{charte2015MLSMOTE,
  title={MLSMOTE: Approaching imbalanced multilabel learning through synthetic instance generation},
  author={Charte, Francisco and Rivera, Antonio J and del Jesus, Mar{\'\i}a J and Herrera, Francisco},
  journal={Knowledge-Based Systems},
  volume={89},
  pages={385--397},
  year={2015},
  publisher={Elsevier}
}
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def IRLbl(Y):
    # imbalance ratio per label
    pos_nums_per_label = np.sum(Y, axis=0)
    max_pos_nums = np.max(pos_nums_per_label)
    return max_pos_nums / pos_nums_per_label


def MeanIR(Y):
    # average imbalance ratio
    IRLbl_VALUE = IRLbl(Y)
    return np.mean(IRLbl_VALUE)


def TailLabel(Y):
    n, m = Y.shape
    irlbl = IRLbl(Y)
    mean_ir = MeanIR(Y)
    return np.where(irlbl>=mean_ir)[0]


def MinBag(X, Y, label_index):
    pos = np.where(Y[:,label_index]==1)
    sample_index = list(set(pos[0]))
    X_minor, Y_minor = X[sample_index,:], Y[sample_index,:]
    return X_minor, Y_minor


def NN_index(X, k=5):
    # n_neighbors including the sample itself, 
    # so we take the number of n_neighbors as k+1 (as the following shows),
    # then delete itself from the neighbors.
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean', 
                           algorithm='auto').fit(X)
    euclidean, index = nn.kneighbors(X)
    return index[:,1:]


def MLSMOTE(X_minor, Y_minor, k=5):
    n, p = X_minor.shape
    m = Y_minor.shape[1]
    X_synth = np.zeros((n,p))
    Y_synth = np.zeros((n,m))
    
    nn_index = NN_index(X_minor, k=5)
    for i in range(n):
        # generate sample feature, that is, X
        sample_X = X_minor[i,:]
        rand_ind = np.random.randint(0, k)
        ref_index = nn_index[i,rand_ind]
        refNeigh = X_minor[ref_index,:]
        diff = sample_X - refNeigh
        offset = diff*np.random.uniform(0, 1, (1,p))
        X_synth[i,:] = sample_X + offset
        
        # generate sample labels Y with the voting method
        sample_nn_index = nn_index[i,:]
        nn_label = Y_minor[sample_nn_index,:]
        Y_synth[i,:] = (np.sum(nn_label, axis=0)>=((k+1)/2))
    X_new = np.r_[X_minor, X_synth]
    Y_new = np.r_[Y_minor, Y_synth]
    return X_new, Y_new
