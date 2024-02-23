import numpy as np
import os
np.random.seed(42)

folder_name = 'bars_and_stripes'
os.makedirs(folder_name, exist_ok=True)

#image size
width = 16
height = 16

n_samples = 50

#noise standard deviation
noise_std = 0.5

####################################################

def gen_data(N):
    X = np.ones([N, 1, height, width]) * -1
    y = np.zeros([N])

    for i in range(len(X)):
        if np.random.rand() > 0.5:
            rows = np.where(np.random.rand(width) > 0.5)[0]
            X[i, 0, rows, :] = 1.
            y[i] = -1
        else:
            columns = np.where(np.random.rand(height) > 0.5)[0]
            X[i, 0, :, columns] = 1.
            y[i] = +1
        X[i, 0] = X[i, 0] + np.random.normal(0, noise_std, size=X[i, 0].shape)

    return X,y

X, y = gen_data(n_samples)

np.savetxt(f'bars_and_stripes/bars_and_stripes_{height}_x_{width}_{noise_std}noise.csv',
                   np.c_[np.reshape(X,[n_samples,-1]), y], delimiter=',')

