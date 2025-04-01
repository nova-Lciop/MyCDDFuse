import torch
print(torch.cuda.is_available())
from kornia.losses import SSIMLoss
print(SSIMLoss)