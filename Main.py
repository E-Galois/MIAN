import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from setting import *
from MICH import MICH
import torch
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = MICH()
    model.train()
