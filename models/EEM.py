
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEM(nn.Module):
    def __init__(self):
        super(EEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm after conv1

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # BatchNorm after conv2

        
    def forward(self, xyzrgb):

        l1 = F.relu(self.bn1(self.conv1(xyzrgb)), inplace=True)  
        l2 = self.bn2(self.conv2(l1))

        return l2
