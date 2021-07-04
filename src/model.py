import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

class ResBlock(nn.Module):
    
    def __init__(self, input_size, output_size, stride, downsample):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, bias=False,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(output_size)
        self.conv2 = nn.Conv2d(output_size, output_size, bias=False,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_size)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_sizes):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_size, hidden_sizes[0], bias=False,
                               kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(hidden_sizes[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.res1 = self.make_block(hidden_sizes[0], hidden_sizes[1], 1)
        self.res2 = self.make_block(hidden_sizes[1], hidden_sizes[2], 2)
        self.res3 = self.make_block(hidden_sizes[2], hidden_sizes[3], 2)
        self.res4 = self.make_block(hidden_sizes[3], hidden_sizes[4], 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_sizes[4], output_size)
        
    def make_block(self, input_size, output_size, stride):
        downsample = None if stride == 1 else nn.Sequential(
            nn.Conv2d(input_size, output_size, bias=False,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_size)
        )
        return nn.Sequential(
            ResBlock(input_size, output_size, stride, downsample),
            ResBlock(output_size, output_size, 1, None)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        out = self.avgpool(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out

class MNISTClassifier(pl.LightningModule):
    
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.resnet = ResNet(1, 10, [64, 64, 128, 256, 512])
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        
    def forward(self, x):
        return self.resnet(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.train_acc(y_hat.softmax(dim=-1), y)
        loss = F.cross_entropy(y_hat, y)
        self.log('loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.val_acc(y_hat.softmax(dim=-1), y)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-4)