import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


class BNN(nn.module):
    def __init__(self):
        self.l1 = 