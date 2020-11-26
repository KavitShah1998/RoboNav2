import sys
import os

print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
log_dir = './logs_1/'
tensorboard = SummaryWriter(log_dir=log_dir)
for i in range(50):
	tensorboard.add_scalar('my_variable',i,i)

tensorboard.close()