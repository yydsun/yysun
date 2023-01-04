#Jupyter Notebook
import math
import cmath
import turtle
'''
name = input("Please input your name:")
print('hello',name)

a = int(input('please input a number:'))
b = int(input('please input another number:'))

print('{} + {} = {}'.format(a,b,a+b))
print('{0} * {0} * {0} = {1}'.format(a,a*a*a))
print(f'{a} * {a} * {a} = {a*a*a}')

c = float(input('please input a number:'))
'''

'''
print(2**32)
print(0b11111111)
print(0xff)
print(0o777)
print(bin(11))
print(hex())#16进制
print(oct())#8进制数字
print(pow(2,3))#幂值
print(round(3/4))#四舍五入
print(math.floor(3/4))#取小
print(math.sqrt())#开方
print(cmath.sqrt(-1))#虚数运算
'''

'''
if 3 == 4:
    print("I guess I'm dreaming")
else:
    print('Get up and go to work')

x = 1
while x < 10:
    print(x, end = '\n')
    x = x+1
'''

'''
#绘图
turtle.reset()
turtle.hideturtle()
turtle.speed(0.05)
turtle.bgcolor('black')
turtle.color(1.0,1.0,1.0)

x = 0
while x < 400:
    turtle.forward(x)
    turtle.right(72) #转x°角
    x = x + 2
turtle.exitonclick()
'''
#字符串不可变
s = "hello"
print(s[1])

#repr是存储的样子
long_string = '''hello
world'''
print(repr(long_string))