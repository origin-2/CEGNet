
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEM(nn.Module):
    def __init__(self):
        super(EEM, self).__init__()

        self.model1 = Block_1()
        self.model2 = Block_2()

    def forward(self, xyzrgb):

        output1 = self.model1(xyzrgb)
        output2 = self.model2(xyzrgb)

        l2 = torch.cat((output1, output2), dim=1)

        return l2



class Block_1(nn.Module):
    def __init__(self):
        super(Block_1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)


    def forward(self, xyzrgb):

        l1 = F.relu(self.bn1(self.conv1(xyzrgb)), inplace=True)  
        l2_1 = self.bn2(self.conv2(l1))

        return l2_1
    

class Block_2(nn.Module):
    def __init__(self):
        super(Block_2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(128)


        
    def forward(self, xyzrgb):

        l1 = F.relu(self.bn1(self.conv1(xyzrgb)), inplace=True)  
        l2_2 = self.bn2(self.conv2(l1))

        return l2_2 
    
