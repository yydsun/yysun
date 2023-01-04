from base64 import encode
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np


class DoubleConv(nn.Module):
    #两次卷积
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),#kernals=3,步长=1,paddings=1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),#kernals=3,步长=1,paddings=1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )
    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(
            self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512],
    ):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()       
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self,x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x =TF.resize(x, size = skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)
def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
    #如果模块是被直接运行的，则代码块被运行，如果模块是被导入的，则代码块不被运行，防止from import运行不想要的.py
'''
def contracting_block(in_channels, out_channels):
    #下采样
    block = torch.nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )
    return block

class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size = (3,3), stride = 2,
                                     padding = 1, output_padding = 1) #右侧过程上采样，stride=2时图像大小翻倍, out_padding = 1
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size = (3,3), in_channels = in_channels, out_channels = mid_channels),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size = (3,3), in_channels = mid_channels, out_channels = out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, e, d):
        d = self.up(d)
        #切割上下采样过程中尺寸不同的部分
        diffY = e.size()[2] - d.size()[2]
        diffX = e.size()[3] - d.size()[3]
        e = e[:,:, diffY//2:e.size()[2]-diffY//2, diffX//2:e.size()[3]-diffX//2]
        cat = torch.cat([e, d], dim = 1)
        out = self.block(cat)
        return out

def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )
    return block

class UNet(nn.Module):
    def __init__(self, in_channel , out_channel):
        super(UNet, self).__init__()
        #下采样
        self.conv_encode1 = contracting_block(in_channels = in_channel, out_channels = 64)
        self.conv_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_encode2 = contracting_block(in_channels = 64, out_channels = 128)
        self.conv_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_encode3 = contracting_block(in_channels = 128, out_channels = 256)
        self.conv_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_encode4 = contracting_block(in_channels = 256, out_channels = 512)
        self.conv_pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        #Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size = (3,3), in_channels = 512, out_channels = 1024),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size = (3,3), in_channels = 1024, out_channels = 1024),
            nn.ReLU(),
            nn.BatchNorm2d(1024)
        )
        #上采样
        self.conv_decode4 = expansive_block(1024,512,512)
        self.conv_decode3 = expansive_block(512,256,256)
        self.conv_decode2 = expansive_block(256,128,128)
        self.conv_decode1 = expansive_block(128,64,64)
        self.final_layer = final_block(64,out_channel)
    def forward(self, x):
        #下采样
        encode_block1 = self.conv_encode1(x) ; print('encode_block1:', encode_block1.size())
        encode_pool1 = self.conv_pool1(encode_block1) ; print('encode_pool1:', encode_pool1.size())
        encode_block2 = self.conv_encode2(encode_pool1) ; print('encode_block2:', encode_block2.size())
        encode_pool2 = self.conv_pool2(encode_block2) ; print('encode_pool2:', encode_pool2.size())
        encode_block3 = self.conv_encode3(encode_pool2) ; print('encode_block3:', encode_block3.size())
        encode_pool3 = self.conv_pool3(encode_block3) ; print('encode_pool3:', encode_pool3.size())
        encode_block4 = self.conv_encode4(encode_pool3) ; print('encode_block4:', encode_block4.size())
        encode_pool4 = self.conv_pool4(encode_block4) ; print('encode_pool4:', encode_pool4.size())

        #bottleneck
        bottleneck = self.bottleneck(encode_pool4) ; print('bottleneck:', bottleneck.size())

        #上采样
        decode_block4 = self.conv_decode4(encode_block4, bottleneck) ; print('decode_block4:', decode_block4.size())
        decode_block3 = self.conv_decode3(encode_block3, decode_block4) ; print('decode_block3:', decode_block3.size())
        decode_block2 = self.conv_decode2(encode_block2, decode_block3) ; print('decode_block2:', decode_block2.size())
        decode_block1 = self.conv_decode1(encode_block1, decode_block2) ; print('decode_block1:', decode_block1.size())

        final_layer = self.final_layer(decode_block1)
        return final_layer

#数据集
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import cv2
import torch
import numpy as np
import SimpleITK as sitk


class Dataset(Dataset):
    def __init__(self,train):
        if train:
            self.datapath = {'images': '../Data/dataset/train.txt', 'annotations':'../Data/dataset/train.txt'}
            #只获取了名称，还需要加0000或0003
        else:
            self.datapath = {'images': '../Data/dataset/text.txt', 'annotations':'../Data/dataset/text.txt'}
        self.image_list, self.target_list = self.read_txt(self.datapath)
    

#两个读取数据的函数，read_txt、read_json
    def read_txt(self,datapath):
        im =[]
        target_image = []
        print(datapath)
        with open(datapath['image'], 'r') as f:
            images_list = f.readlines()
            images_list = [f'{word} 0000' for word in images_list]
        with open(datapath['target'], 'r') as f:
            annotations_list = f.readlines()
            annotations_list = [f'{word} 0003' for word in annotations_list]
        return images_list, annotations_list
    def load_images(dir_path, dtype=np.float32):
        
        for file_path in dir:
            image = sitk.ReadImage(file_path)
            data = np.asarray(sitk.GetArrayFromImage(image), dtype=dtype)
    
    def read_json(save_path, encoding='utf8'):
        jsondata = []
        with open(save_path, 'r', encoding=encoding) as f:
            content = f.read()
            content = json.loads(content)
            for key in content:
                jsondata.append(content[key])
            return jsondata
    
    def __getitem__(self, item):
        # 最核心的部分,经过处理,要返回输入和gt

        return img, target

    def __len__(self):
		# 这可以根据具体情况修改，不写也行
        return len(self.data)
    
'''
