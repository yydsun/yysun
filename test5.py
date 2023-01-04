import os
import matplotlib.pyplot as plt
import torchcam
import pandas as pd

from PIL import Image

# windows操作系统
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号\

# ImageNet1000类别名称与ID号
df = pd.read_csv('imagenet_class_index.csv')
python torch-cam/scripts/cam_example.py \
    #文件
        --img test_img/border-collie.jpg \
            #图片
        --savefig output/B1_border_collie.jpg \
            #保存路径
        --arch resnet18 \
            #模型结构
        --class-idx 232 \
            #类别
        --rows 2
        #两行图片


# 类别-虎斑猫
!python torch-cam/scripts/cam_example.py \
        --img test_img/cat_dog.jpg \
        --savefig output/B2_cat_dog.jpg \
        --arch resnet18 \
        --class-idx 282 \
        --rows 2