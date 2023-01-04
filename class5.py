#sigmoid

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    #概率趋近于0，1的时候学不动了
'''
x = np.arange(-10, 10, 0.1)
y = sigmoid(x)
plt.plot(x, y, linewidth = 3)
#linewidth 线宽
plt.plot([-10 ,10], [1, 1], 'b--')
plt.xlabel('z')
plt.ylabel('h(z)')
plt.show()
print()
'''
#交叉熵
#comparing sum of square loss with cross-entropy loss
#when used with sigmoid function

np.random.seed(0)
x = 10 * np.random.rand(50)
y = np.zeros_like(x)
y[x>5] = 1
pos_index = np.where(x>5)[0]
neg_index = np.where(x<=5)[0]

ax = plt.axes()
ax.scatter(x[pos_index], y[pos_index], marker = 'o')
ax.scatter(x[neg_index], y[neg_index], marker = '+')
plt.show()

h = lambda theta: sigmoid(x - theta)
sos_loss = lambda theta: np.square(y - h(theta)).sum() / x.size
ce_loss = lambda theta: -1 * (np.log10(h(theta)) * y + np.log10(1-h(theta))*(1-y)).sum() / x.size

sos_losses = []
ce_losses = []
theta_list = np.arange(-5, 15, 0.1)
for theta in theta_list:
    sos_losses.append(sos_loss(theta))
    ce_losses.append(ce_loss(theta))
plt.plot(theta_list, sos_losses, 'r--')
plt.plot(theta_list, ce_losses)
plt.show()