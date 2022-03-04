from setting import *


def calc_neighbor(label_1, label_2):
    Sim = (np.dot(label_1, label_2.transpose()) > 0).astype(np.float32)*0.999
    return Sim


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr