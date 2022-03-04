import numpy as np
import scipy.io
from load_data import loading_data
from load_data import split_data

# environment and parameters
checkpoint_path = './checkpoint'

SEMANTIC_EMBED = 512
MAX_ITER = 100
batch_size = 64

images, tags, labels = loading_data()
dimTxt = tags.shape[1]
dimLab = labels.shape[1]

DATABASE_SIZE = 18015
TRAINING_SIZE = 18015
QUERY_SIZE = 2000

X, Y, L = split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE)
train_L = L['train']
train_x = X['train']
train_y = Y['train']

query_L = L['query']
query_x = X['query']
query_y = Y['query']

retrieval_L = L['retrieval']
retrieval_x = X['retrieval']
retrieval_y = Y['retrieval']

num_train = train_x.shape[0]
numClass = train_L.shape[1]

Sim = (np.dot(train_L, train_L.transpose()) > 0).astype(int)*0.999

Epoch = 30
k_lab_net = 10
k_img_net = 15
k_txt_net = 15

bit = 16
# hyper here

# Learning rate
lr_lab = [np.power(0.1, x) for x in np.arange(2.0, MAX_ITER, 0.5)]
lr_img = [np.power(0.1, x) for x in np.arange(4.5, MAX_ITER, 0.5)]
lr_txt = [np.power(0.1, x) for x in np.arange(3.5, MAX_ITER, 0.5)]
lr_dis = [np.power(0.1, x) for x in np.arange(3.0, MAX_ITER, 0.5)]

