import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.module):
	def __init__(self, input_size, hidden_size, output_size):
