import numpy as np
import matplotlib.pyplot as plt
import math

def create_data(theta0 = 1,theta1 = 1, size = 50):
    x = 4 * np.random.rand(size)
    noise = 0.3 * np.random.randn(size)
    #加噪声，可以改
    y = theta1 * x + theta0 + noise
    return x, y

x,y = create_data(1,1,50)
fig1 = plt.figure()
fig1.gca().scatter(x, y)
fig1.show()

def train(loss, gradient_theta0, gradient_theta1, alpha = 0.05, iteration = 250):
    #iteration迭代次数
    total_loss = []
    theta0, theta1 = 0., 0.
    for _ in range(iteration):
        current_loss = loss(theta0, theta1)
        g0 = gradient_theta0(theta0, theta1)
        g1 = gradient_theta1(theta0, theta1)
        total_loss.append(math.log(current_loss))

        theta0 -= alpha * g0
        #梯度下降
        theta1 -= alpha * g1
    return theta0, theta1, total_loss

loss = lambda t0, t1: 0.5*np.square(y - (x*t1 + t0)).sum() / x.size
#lambda函数的两个参数
gradient_theta1 = lambda t0, t1: -((y-(x*t1 + t0))*x).sum() / x.size
#偏导
gradient_theta0 = lambda t0, t1: -(y - (x*t1 + t0)).sum() / x.size

theta0, theta1, total_loss = train(loss, gradient_theta0, gradient_theta1, 0.05)
print(f'theta0={theta0} theta1={theta1} loss={total_loss[-1]}')

fig1 = plt.figure()
fig1.gca().scatter(x,y)
x1 = [i * 4 / 50 for i in range(0,50)]
#散点图
y_hat = [theta1 * x + theta0 for x in x1]
#计算预测值
fig1.gca().plot(x1, y_hat)
#连线
fig1.gca().set(xlabel = 'x', ylabel = 'y')
fig1.show()

fig2 = plt.figure()
ax = fig2.gca()
ax.plot(total_loss)
ax.set(xlabel = 'iteration', ylabel = 'loss')
fig2.show()
input()

