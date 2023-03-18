# from numpy import identity
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,80,3,1,1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(80,400,3,1,1),
            nn.BatchNorm2d(400),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(64,64,3,1,1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )

        #dropout可以防过拟合
        # self.dropout1 = 0.2
        # self.dropout2 = 0.5

        self.FC = nn.Sequential(
            nn.Linear(400 * 4 * 4, 1800),
            nn.ReLU(),
            # nn.Dropout(self.dropout1),
            nn.Linear(1800, 400),
            nn.ReLU(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 80),
            nn.ReLU(),
            nn.Linear(80,10)
        )
    
    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x) 
        # x = self.conv4(x)
        # print(x.shape)
        # exit(0)
        x = x.view(x.size(0),-1)
        x = self.FC(x)
        return x

