#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.Res2Net_v1b import res2net50_v1b_26w_4s
from module.BasicConv2d import BasicConv2d
from module.SA import WeightedGate
class BMM(nn.Module):
    def __init__(self, channels):
        super(BMM, self).__init__()
        self.cbr_f1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))    
        self.cbr_f2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))        
        self.cbr_f5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels))
        self.WG = WeightedGate()
        self.convup1 = nn.Upsample(scale_factor=2, 
                                   mode='bilinear', 
                                   align_corners=True)
        self.convup2 = nn.Upsample(scale_factor=2, 
                                   mode='bilinear', 
                                   align_corners=True) 
        self.convup3 = nn.Upsample(scale_factor=2, 
                                   mode='bilinear', 
                                   align_corners=True) 
        self.convup4 = nn.Upsample(scale_factor=2, 
                                   mode='bilinear', 
                                   align_corners=True)
        self.convup5 = nn.Upsample(scale_factor=2, 
                                   mode='bilinear', 
                                   align_corners=True)
        self.convd1 = nn.Conv2d(3,3,kernel_size=2,stride=2)
        self.convd2 = nn.Conv2d(3,3,kernel_size=2,stride=2)
        self.convd3 = nn.Conv2d(3,3,kernel_size=2,stride=2)
        self.convd4 = nn.Conv2d(3,3,kernel_size=2,stride=2)

    def forward(self, f1, f2, f5):
        f1_out = self.cbr_f1(f1)
        f2 = self.convup1(f2)
        f2_out = self.cbr_f1(f2)
        f2_branch1 = self.WG(f2_out+f1_out)
        f2_branch2 = self.mlp(f2_branch1)
        f2_branch3 =self.sigmoid(f2_branch1)
        f51 = self.convup2(f5)
        f52 = self.convup3(f51)
        f53 = self.convup4(f52)
        f54 = self.convup5(f53)
        f5_out = self.cbr_f1(f54)
        f5_branch2 = self.mlp(f5_out)
        fb = f5_branch2 * f2_branch3
        fb1 = self.convd1(fb)
        fb2 = self.convd2(fb1)
        fb3 = self.convd3(fb2)
        fb4 = self.convd4(fb3)
        return fb4        
    def initialize(self):
        weight_init(self)
class fsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(fsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
    def initialize(self):
        weight_init(self)
class fSwish(nn.Module):
    def __init__(self, inplace=True):
        super(fSwish, self).__init__()
        self.sigmoid = fsigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)
    def initialize(self):
        weight_init(self)
class BEM(nn.Module):
    def __init__(self, channels):
        super(BEM, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, 
                      channels, 
                      kernel_size=3, 
                      padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 
                      kernel_size=3, 
                      dilation=3, 
                      padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.LSwish = fSwish()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, 
                                     stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, 
                                     stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.convupb = nn.Upsample(scale_factor=2,
                                   mode='bilinear', 
                                   align_corners=True)
    def forward(self, x):
        conv = self.convupb(x)
        conv_out = self.conv_block(conv)
        atrous_out = self.atrous_conv(conv_out)
        atrous_branch1 = self.avg_pool(atrous_out)
        softmax_out = self.softmax(atrous_branch1)
        max_pooled = self.max_pool(atrous_out)
        softmax_out1 = self.softmax(max_pooled)
        atrous_branch2 = softmax_out1 * softmax_out
        conv_swish_out = self.LSwish(conv_out)
        global_avg_pooled = self.global_avg_pool(conv_swish_out)
        fbemas =  atrous_out + atrous_branch2 + global_avg_pooled
        fbems = self.sigmoid(fbemas)
        fbem = self.conv1(fbems)
        return fbem   
    def initialize(self):
        weight_init(self)
class FFM(nn.Module):
    def __init__(self, channels):
        super(FFM, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 
                               kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 
                               kernel_size=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 
                               kernel_size=1, padding=1)
        self.atrous_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 
                      kernel_size=3, dilation=3, padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.atrous_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, 
                      dilation=5, padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.atrous_conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, 
                      dilation=7, padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.avg_pool = nn.AvgPool2d(kernel_size=2, 
                                     stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.convupf = nn.Upsample(scale_factor=2,
                                   mode='bilinear', 
                                   align_corners=True)
    def forward(self, f1, f2, f3):
        f1 = self.atrous_conv1(f1)
        f2 = self.atrous_conv2(f2)
        f3 = self.atrous_conv3(f3)
        f_add1 = f1 + f2
        gap1 = self.global_avg_pool(f_add1)
        fs = self.softmax(gap1)
        f_conv1 = self.conv1(fs)
        gap2 = self.sigmoid(f_conv1)
        fs3 = self.convupf(f3)
        f_add2 = f2+ fs3
        f_conv2 = self.conv2(f_add2)
        gap = self.avg_pool(f_conv2)
        gap = self.conv3(gap)
        ffs =  gap + gap2    
        fffm =  self.sigmoid(ffs)
        return fffm
    def initialize(self):
        weight_init(self)

class BASNet(nn.Module):
    def __init__(self):
        super(BASNet, self).__init__()
        self.backbone = res2net50_v1b_26w_4s()
        self.ffm1 = FFM(64)
        self.ffm2 = FFM(128)
        self.ffm3 = FFM(256)
        self.ffm4 = FFM(512)  
        self.bem1 = BEM(256)
        self.bem2 = BEM(128)
        self.bem3 = BEM(64)
        self.bem4 = BEM(32)
        self.bmm = BMM(32)
        self.output_conv = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.edge_conv = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.upsample2_conv = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.upsample4_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.upsample8_conv = BasicConv2d(128, 128, kernel_size=3, padding=1)
        self.upsample16_conv = BasicConv2d(256, 256, kernel_size=3, padding=1)
        self.upsample32_conv = BasicConv2d(512, 512, kernel_size=3, padding=1)
    def forward(self, x):
        image_size = x.size()[2:]
        x_backbone = self.backbone.conv1(x)
        x_backbone = self.backbone.bn1(x_backbone)
        x_backbone = self.backbone.relu(x_backbone)
        x_backbone = self.backbone.maxpool(x_backbone)
        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        layer5 = self.backbone.layer5(layer4)  
        bmm1 = self.bmm(layer1,layer2,layer5)
        bmm2= self.linearf1(bmm1) + layer5
        b4 = self.bem4(bmm2)
        b3 = self.bem3(b4)
        b2 = self.bem2(b3)
        b1 = self.bem1(b2)
        f4 = self.ffm1(layer4,b4,bmm2)
        f3 = self.ffm2(layer3,b3,f4)
        f2 = self.ffm3(layer2,b2,f3)
        f1 = self.ffm4(layer1,b1,f2)
        fu5 = self.upsample2_conv(bmm2)
        fu4 = self.upsample4_conv(fu5)
        fs4 = fu4 + f4
        fu3 = self.upsample8_conv(fs4)
        fs3 = fu3 + f3
        fu2 = self.upsample16_conv(fs3)
        fs2 = fu2 + f2
        fu1 = self.upsample32_conv(fs2)
        fs1 = fu1 + f1
        map = self.output_conv(fs1)
        output = F.interpolate(map, size=image_size, mode='bilinear')
        edge_features = self.bmm(layer5)
        edge_map = self.edge_conv(edge_features)
        edge_map = F.interpolate(edge_map, size=image_size, mode='bilinear')   
        return output, edge_map
    def initialize(self):
        weight_init(self)
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is None:
                pass
            elif m.bias is not None:
                nn.init.zeros_(m.bias)
            else:
                nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Upsample, Parameter, nn.AdaptiveAvgPool2d, nn.Sigmoid)):
            pass
        else:
            m.initialize()
