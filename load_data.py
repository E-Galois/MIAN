import h5py
import numpy as np
from scipy.io import loadmat


def loading_data():
    file = h5py.File('./data/Flicker/IAll/mirflickr25k-iall.mat')
    images = (file['IAll'][:].transpose(0, 1, 3, 2) / 255.0).astype(np.float32)
    tags = loadmat('./data/Flicker/YAll/mirflickr25k-yall.mat')['YAll'].astype(np.float32)
    labels = loadmat('./data/Flicker/LAll/mirflickr25k-lall.mat')['LAll'].astype(np.float32)
    file.close()
    return images, tags, labels


def split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE):
    X = {}
    index_all = np.random.permutation(QUERY_SIZE+DATABASE_SIZE)
    ind_Q = index_all[0:QUERY_SIZE]
    ind_T = index_all[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE]
    ind_R = index_all[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE]

    X['query'] = images[ind_Q, :, :, :]
    X['train'] = images[ind_T, :, :, :]
    X['retrieval'] = images[ind_R, :, :, :]

    Y = {}
    Y['query'] = tags[ind_Q, :]
    Y['train'] = tags[ind_T, :]
    Y['retrieval'] = tags[ind_R, :]

    L = {}
    L['query'] = labels[ind_Q, :]
    L['train'] = labels[ind_T, :]
    L['retrieval'] = labels[ind_R, :]
    return X, Y, L
