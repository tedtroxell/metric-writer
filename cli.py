from src.configs.interface import DefaultConfigInterface
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(32,32),
    nn.Tanh(),
    nn.Linear(32,1)
)

print(DefaultConfigInterface.auto_config( model ).config_type)