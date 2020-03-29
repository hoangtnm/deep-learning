import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 224x224x3 image tensor)
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(24, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(64*3*3, 144)
        self.fc2 = nn.Linear(144, 1)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        # flatten image input
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
