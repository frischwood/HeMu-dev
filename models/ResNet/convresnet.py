import torch
import torch.nn as nn
from torch.autograd import Variable 

###
# 2D ResNet
###
class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())
            self.conv2 = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                            nn.BatchNorm2d(out_channels))
            self.downsample = downsample
            self.relu = nn.ReLU()
            self.out_channels = out_channels

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out
    
class ConvResNet(nn.Module):
        """Schuurman's resnet"""
        def __init__(self, config):
            block = ResidualBlock
            layers =  [3,3,6,3]
            in_channels = int(config["num_channels"] + 1) # + 1 for added hod channel
            out_channels = config["num_classes"]

            super(ResNet, self).__init__()
            self.inplanes = 64
            self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, self.inplanes, kernel_size = 1, stride = 1, padding = 0),
                            nn.BatchNorm2d(64),
                            nn.ReLU())
            self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
            self.layer0 = self._make_layer(block, 128, layers[0], stride = 1)
            self.layer1 = self._make_layer(block, 256, layers[1], stride = 1)
            self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
            self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding = 0)
            
            self.gap = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(3) # size of img 2x (2x2) max pooled 
            )
            # self.fc = nn.Linear(512, num_classes)
            self.fc = nn.Linear(256, out_channels)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes:

                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(planes),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.layer0(x)
            x = self.maxpool(x)
            x = self.layer1(x)

            # x = self.layer2(x)
            # x = self.layer3(x)

            # x = self.avgpool(x)
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

