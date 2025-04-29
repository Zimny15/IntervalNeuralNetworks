import torch
import torch.nn as nn
from cnn import DeconvNet

base_model = DeconvNet()
base_model.load_state_dict(torch.load("base_model.pth"))

for m in base_model.modules():
    print(m)