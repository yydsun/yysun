import lime
import sklearn
import os
"""
import wget
1
# 存放测试图片
os.mkdir('test_img')

# 存放模型权重文件
os.mkdir('checkpoint')

wget.download('https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/checkpoints/fruit30_pytorch_20220814.pth', "C:\complaire\data\lime") 
wget.download('https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/fruit30/labels_to_idx.npy',  "C:\complaire\data\lime")
wget.download('https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/fruit30/idx_to_labels.npy',  "C:\complaire\data\lime")
wget.download('https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/cat_dog.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_orange_2.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_bananan.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/0818/test_草莓.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_石榴.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_orange.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_lemon.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_火龙果.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/watermelon1.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/test/banana1.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_fruits.jpg', 'test_img')
wget.download('https://zihao-openmmlab.obs.myhuaweicloud.com/20220716-mmclassification/test/0818/test_kiwi.jpg', 'test_img')

2

import numpy as np
import pandas as pd

import lime
from lime import lime_tabular

df = pd.read_csv('wine.csv')
print(df.shape)
from sklearn.model_selection import train_test_split

#重要！划分数据集方式

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#训练测试特征、训练测试标签
'''
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
'''
from sklearn.ensemble import RandomForestClassifier
#随机森林算法
model = RandomForestClassifier(random_state=42)
#训练集的特征和标签上训练模型
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
#评估模型
print(score)

#初始化LIME可解释性分析算法
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train), # 训练集特征，必须是 numpy 的 Array
    feature_names=X_train.columns, # 特征列名
    class_names=['bad', 'good'], # 预测类别名称
    mode='classification' # 分类模式
)
#测试集中选取一个样本，输入训练好的模型中预测
data_test = np.array(X_test.iloc[idx]).reshape(1, -1)
prediction = model.predict(data_test)[0]
y_true = np.array(y_test)[idx]
print('测试集中的 {} 号样本, 模型预测为 {}, 真实类别为 {}'.format(idx, prediction, y_true))

#可解释性分析的部分，输出
exp = explainer.explain_instance(
    data_row=X_test.iloc[idx], 
    predict_fn=model.predict_proba
)

exp.show_in_notebooks(show_table=True)
"""
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

img_path = 'test_img/cat_dog.jpg'
img_pil = Image.open(img_path)

model = models.inception_v3(pretrained=True).eval().to(device)
idx2label, cls2label, cls2idx = [], {}, {}
with open(os.path.abspath('imagenet_class_index.json'), 'r') as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))} 
trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

trans_A = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    trans_norm
    ])

trans_B = transforms.Compose([
        transforms.ToTensor(),
        trans_norm
    ])

trans_C = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224)
])
input_tensor = trans_A(img_pil).unsqueeze(0).to(device)
pred_logits = model(input_tensor)
pred_softmax = F.softmax(pred_logits, dim=1)
top_n = pred_softmax.topk(5)

print(top_n)
def batch_predict(images):
    batch = torch.stack(tuple(trans_B(i) for i in images), dim=0)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()
test_pred = batch_predict([trans_C(img_pil)])
test_pred.squeeze().argmax()   
from lime import lime_image
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(trans_C(img_pil)), 
                                         batch_predict, # 分类预测函数
                                         top_labels=5, 
                                         hide_color=0, 
                                         num_samples=8000) # LIME生成的邻域图像个数
from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=20, hide_rest=False)
img_boundry = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry)
plt.show()
temp, mask = explanation.get_image_and_mask(281, positive_only=False, num_features=20, hide_rest=False)
img_boundry = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry)
plt.show()                                     