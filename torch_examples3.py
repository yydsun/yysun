import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1,2,3])
#包含单一数据类型元素的多维矩阵
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3,1),#nn.Linear需要输入两个参数，in_features为上一层神经元的个数（输入），out_features为这一层的神经元个数（输出）
    torch.nn.Flatten(0,1)#降维，torch.flatten(x)等于torch.flatten(x，0)默认将张量拉成一维的向量，也就是说从第一维开始平坦化，torch.flatten(x，1)代表从第二维开始平坦化
)
loss_fn = torch.nn.MSELoss(reduction = 'sum')
#均方误差，loss = (x - y)^2
learning_rate = 1e-6
for t in range(2000):
    y_pred = model(xx)
    
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    model.zero_grad()
    #梯度清零
    loss.backward()

    with torch.no_grad():#param允许梯度计算，若没有torch.no_grad()则会在计算过程中自动构建计算图，产生不必要的显存占用
        for param in model.parameters():
            param -= learning_rate * param.grad
            # 你也可以就像得到列表（list）的第一个元素一样，得到 model 的第一层

linear_layer = model[0]
# 对于 linear layer，它的参数被存储为 weight 和 bias。

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

