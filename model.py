import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, numClasses):
        super(Model, self).__init__() #make the model inherit from the nn.Module

        self.conv1 = nn.Conv2d(13, 64, kernel_size=4, padding=1) #make the first cnn layer. this one goes from 13 (13 board states) to 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, numClasses)
        self.relu = nn.ReLu()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x