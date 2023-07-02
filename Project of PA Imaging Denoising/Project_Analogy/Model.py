import torch
import torch.nn as nn
import torch.utils.data

class DoubleConv(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.doubleconv=nn.Sequential(
            nn.Conv1d(in_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Conv1d(out_c,out_c,kernel_size=3,padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU()
        )

    def forward(self,x):
        return self.doubleconv(x)

class Down(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.down=nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_c,out_c)
        )

    def forward(self,x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv=DoubleConv(in_c,out_c)

    def forward(self,x1,x2):
        x1=self.up(x1)
        x=torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self,class_num,channel_num):
        super().__init__()
        self.conv1=DoubleConv(channel_num,16)
        self.down1=Down(16,32)
        self.down2=Down(32,64)
        self.down3=Down(64,128)
        self.down4=Down(128,128)
        self.up1=Up(256,64)
        self.up2=Up(128,32)
        self.up3=Up(64,16)
        self.up4=Up(32,16)
        self.out=nn.Conv1d(16,class_num,kernel_size=1)

    def forward(self,x):
        x1=self.conv1(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)
        x=self.up1(x5,x4)
        x=self.up2(x,x3)
        x=self.up3(x,x2)
        x=self.up4(x,x1)
        x=self.out(x)
        return x