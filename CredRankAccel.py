import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from numba import cuda, float32, guvectorize, njit




@njit
def jaccard(x, y):
     #x = np.asarray(x)
     #y = np.asarray(y)
     xt = x != 0
     yt = y != 0
     num = np.sum(np.logical_xor(xt, yt).astype(np.int32))
     denom = np.sum(np.logical_or(np.logical_and(xt, yt), np.logical_xor(xt, yt)).astype(np.int32))
     return np.where(denom == 0, 0.0, num.astype(np.float32) / denom)

@guvectorize([(float32[:, :], float32[:, :])],'(n_users, n_news), (n_users, n_users) -> ()', nopython=True)
def getSigma(X, sigma):
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            xt = X[i] != 0
            yt = X[j] != 0
            num = np.sum(np.logical_xor(xt, yt).astype(np.float32))
            denom = np.sum(np.logical_or(np.logical_and(xt, yt), np.logical_xor(xt, yt)).astype(np.float32))
            sigma[i, j] = np.where(denom == 0, 0.0, num / denom)

def makeCreds(topicname):
    mat = np.load('results/{}_W.npy'.format(topicname))
    U = np.load('results/{}_U.npy'.format(topicname))

    #print(len(mat), len(U))
    m = len(mat)
    sigma = np.zeros((m, m))
    sigma.fill(1)
    #print(mat.shape)
    print(mat.shape[0])
    sigma = getSigma(mat)
    '''
    for i in range(m):
        for j in range(i+1, m):
            #print(i, mat[i])
            #print(j, mat[j])
            score = jaccard_score(mat[i], mat[j], zero_division=0.0)
            #print(score)
            sigma[i, j] = score
    '''

    print(sigma)
    print(sigma[0])
    np.save('results/{}_sigma.npy'.format(topicname), sigma)


if __name__ == '__main__':
    makeCreds("charliehebdo")