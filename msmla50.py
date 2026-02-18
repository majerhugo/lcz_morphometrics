import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from torchvision.transforms import v2 as transforms

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        se = self.global_avg_pool(x)  
        se = F.relu(self.fc1(se))             
        se = torch.sigmoid(self.fc2(se))         
        return x * se                        

class SEConvBlock(nn.Module):
    def __init__(self, in_channels, filters, f, stride=2):
        super(SEConvBlock, self).__init__()
        f1, f2, f3 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=f, stride=1, padding=f // 2)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)

        self.shortcut_conv = nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride)
        self.shortcut_bn = nn.BatchNorm2d(f3)

        self.se = SEBlock(f3)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x))

        out = F.relu(self.bn1(self.conv1(x)))               
        out = F.relu(self.bn2(self.conv2(out)))             
        out = self.bn3(self.conv3(out))                     

        out = self.se(out)                                  

        out += shortcut                                     
        return F.relu(out)                                  

class SEIdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, f):
        super(SEIdentityBlock, self).__init__()
        f1, f2, f3 = filters

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv2 = nn.Conv2d(f1, f2, kernel_size=f, stride=1, padding=f // 2)
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(f3)

        self.se = SEBlock(f3)

    def forward(self, x):
        shortcut = x                               

        out = F.relu(self.bn1(self.conv1(x)))    
        out = F.relu(self.bn2(self.conv2(out)))   
        out = self.bn3(self.conv3(out))        

        out = self.se(out)                          

        out += shortcut                           
        return F.relu(out)                        

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttention, self).__init__()

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        avg_pool = F.adaptive_avg_pool2d(x, 1)
        avg_out = self.shared_mlp(avg_pool) 

        max_pool = F.adaptive_max_pool2d(x, 1)
        max_out = self.shared_mlp(max_pool) 

        scale = self.sigmoid(avg_out + max_out)

        return x * scale 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        assert avg_pool.shape[1] == 1

        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        assert max_pool.shape[1] == 1

        concat = torch.cat([avg_pool, max_pool], dim=1)
        assert concat.shape[1] == 2

        attention_map = self.conv(concat)
        attention_map = self.sigmoid(attention_map)

        return x * attention_map

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=8, spatial_kernel=7):

        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x

class MSMLA50(nn.Module):
    def __init__(self, input_channels, depth, num_classes=17):
        super(MSMLA50, self).__init__()

        self.conv_5x5 = nn.Conv2d(input_channels, 16, kernel_size=5, padding=2)
        self.conv_3x3 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2d(input_channels, 16, kernel_size=1)

        self.cbam_in = CBAM(in_channels=64, reduction=8)

        self.block1 = nn.Sequential(
            SEConvBlock(64, [depth[0], depth[0], depth[0] * 4], f=3, stride=1),
            SEIdentityBlock(depth[0] * 4, [depth[0], depth[0], depth[0] * 4], f=3),
            SEIdentityBlock(depth[0] * 4, [depth[0], depth[0], depth[0] * 4], f=3)
        )
        self.cbam1 = CBAM(depth[0] * 4, reduction=8)

        self.block2 = nn.Sequential(
            SEConvBlock(depth[0] * 4, [depth[1], depth[1], depth[1] * 4], f=3, stride=2),
            SEIdentityBlock(depth[1] * 4, [depth[1], depth[1], depth[1] * 4], f=3),
            SEIdentityBlock(depth[1] * 4, [depth[1], depth[1], depth[1] * 4], f=3),
            SEIdentityBlock(depth[1] * 4, [depth[1], depth[1], depth[1] * 4], f=3)
        )
        self.cbam2 = CBAM(depth[1] * 4, reduction=8)

        self.block3 = nn.Sequential(
            SEConvBlock(depth[1] * 4, [depth[2], depth[2], depth[2] * 4], f=3, stride=2),
            SEIdentityBlock(depth[2] * 4, [depth[2], depth[2], depth[2] * 4], f=3),
            SEIdentityBlock(depth[2] * 4, [depth[2], depth[2], depth[2] * 4], f=3),
            SEIdentityBlock(depth[2] * 4, [depth[2], depth[2], depth[2] * 4], f=3),
            SEIdentityBlock(depth[2] * 4, [depth[2], depth[2], depth[2] * 4], f=3),
            SEIdentityBlock(depth[2] * 4, [depth[2], depth[2], depth[2] * 4], f=3)
        )
        self.cbam3 = CBAM(depth[2] * 4, reduction=8)

        self.fc = nn.Linear(640, num_classes)

    def forward(self, x):
        x0 = self.conv_5x5(x)
        x1 = self.conv_3x3(x)
        x2 = self.conv_1x1(x)
        x3 = torch.cat([x0, x1, x2], dim=1) 
        
        cbam_in = self.cbam_in(x3)
        cbam_in = F.adaptive_avg_pool2d(cbam_in, 1).view(cbam_in.size(0), -1) 

        x = self.block1(x3)
        cbam1 = self.cbam1(x)
        cbam1 = F.adaptive_avg_pool2d(cbam1, 1).view(cbam1.size(0), -1)

        x = self.block2(x)
        cbam2 = self.cbam2(x)
        cbam2 = F.adaptive_avg_pool2d(cbam2, 1).view(cbam2.size(0), -1)

        x = self.block3(x)
        cbam3 = self.cbam3(x)
        cbam3 = F.adaptive_avg_pool2d(cbam3, 1).view(cbam3.size(0), -1) 

        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  

        x = torch.cat([x, cbam_in, cbam1, cbam2, cbam3], dim=1)  

        out = self.fc(x)
        return F.softmax(out, dim=1)

    def get_embedding_raw_fc(self, x):
        x0 = self.conv_5x5(x) 
        x1 = self.conv_3x3(x) 
        x2 = self.conv_1x1(x)
        x3 = torch.cat([x0, x1, x2], dim=1) 

        cbam_in = self.cbam_in(x3)
        cbam_in = F.adaptive_avg_pool2d(cbam_in, 1).view(cbam_in.size(0), -1) 

        x = self.block1(x3)
        cbam1 = self.cbam1(x)
        cbam1 = F.adaptive_avg_pool2d(cbam1, 1).view(cbam1.size(0), -1) 

        x = self.block2(x)
        cbam2 = self.cbam2(x)
        cbam2 = F.adaptive_avg_pool2d(cbam2, 1).view(cbam2.size(0), -1) 

        x = self.block3(x)
        cbam3 = self.cbam3(x)
        cbam3 = F.adaptive_avg_pool2d(cbam3, 1).view(cbam3.size(0), -1)

        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1) 

        x = torch.cat([x, cbam_in, cbam1, cbam2, cbam3], dim=1) 

        return x