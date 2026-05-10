import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



class TargetEncoder(nn.Module):
    def __init__(self,c_in,c_out):
        super().__init__()
        self.target = nn.Sequential(
            ConvBNReLU(c_in,64,stride = 1),
            ConvBNReLU(64,128,stride = 1),
            ConvBNReLU(128,256,stride = 1)
        )

    def forward(self,x):
        return self.target(x)
    
class ContextEncoder(nn.Module): 
    def __init__(self,c_in,c_out):
        super().__init__()
        self.context = nn.Sequential(
            ConvBNReLU(c_in,64,stride = 1),
            ConvBNReLU(64,128,stride = 1),
            ConvBNReLU(128,256,stride = 1)
        )

    def forward(self,x):
        return self.context(x)
    

class ConvBNReLU(nn.Module): 
    def __init__(self,c_in,c_out,stride):
        self.net = nn.Sequential(
            nn.Conv2d(c_in,c_out,stride)
            nn.BatchNorm2d(c_out)
            nn.ReLU()
        )
    def forward(self,x) : 
        return self.net(x)
    

class JEPA(nn.Module): 


