import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score






def makeCreds(topicname):
    mat = np.load('results/{}_W.npy'.format(topicname))
    U = np.load('results/{}_U.npy'.format(topicname))

    #print(len(mat), len(U))
    m = len(mat)
    sigma = np.zeros((m, m))
    sigma.fill(1)
    #print(mat.shape)
    print(mat.shape[0])
    for i in range(m):
        for j in range(i+1, m):
            sigma[i, j] = jaccard_score(mat[i], mat[j], zero_division=0)
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