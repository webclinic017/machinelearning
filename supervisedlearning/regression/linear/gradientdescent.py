# Optimization hàm n biến

# đạo hàm, Đường tiếp tuyến, hệ số góc, cực tiểu, phương trình đạo hàm bằng 0.
# http://giasuttv.net/bang-day-du-cac-cong-thuc-dao-ham-va-meo-hoc-nhanh/

# local minimum: hàm số đạt giá trị nhỏ nhất

# xấp xỉ điểm cần tìm khi số chiều của input và số điểm dữ liệu lớn

# di chuyển ngược/ cùng dấu với đạo hàm -> để điểm ta tìm được gần với cực tiểu

# learning rate

# descent nghĩa là đi ngược, Gradient là đạo hàm

"""
# --------------- coding: Gradient Descent cho hàm 1 biến ----------------
from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt


# f(x) = x**2 + 5*sin(x)
def model_function():
    x = np.linspace(-5, 5, 35)
    y = x**2+5*np.sin(x)
    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(x, y, 'r')

    # show the plot
    plt.show()
    pass


def grad(x):
    return 2*x+5*np.cos(x)


def cost(x):
    return x**2+5*np.sin(x)


def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        # di chuyển ngược
        x_new = x[-1] - eta*grad(x[-1])
        # khi đạo hàm có độ lớn đủ nhỏ
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)


x1, it1 = myGD1(.1, -5)
x2, it2 = myGD1(.1, 5)
print(
    f'Solution x1 = {x1[-1]:.5f}, cost = {cost(x1[-1]):.5f}, obtained after {it1} iterations')
print(
    f'Solution x2 = {x2[-1]:.5f}, cost = {cost(x2[-1]):.5f}, obtained after {it2} iterations')

# learning rate -------------------------------------
# nhỏ η=0.01, tốc độ hội tụ rất chậm

# lớn η=0.5, thuật toán tiến rất nhanh tới
# gần đích sau vài vòng lặp, nhưng không hội tụ được vì bước nhảy quá

# phải thí nghiệm để chọn ra giá trị tốt nhất hoặc:
# chọn learning rate khác nhau ở mỗi vòng lặp
# -------------------------------------

# model_function()
"""

# --------------- coding: Gradient Descent cho hàm nhiều biến ----------------

"""
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
np.random.seed(2)

X = np.random.rand(1000, 1)
# đường thẳng y=4+3x
y = 4 + 3*X + .2*np.random.randn(1000, 1)
# print((X.shape[0], 1))
one = np.ones((X.shape[0], 1))
# print(one)
Xbar = np.concatenate((one, X), axis=1)
# print(Xbar.T)
# link công thức: https://machinelearningcoban.com/2016/12/28/linearregression/
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)

print(f'Solution found by formula: w = {w_lr.T}')

w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

plt.plot(X.T, y.T, 'b.')
plt.plot(x0, y0, 'y', linewidth=2)
plt.axis([0, 1, 0, 10])
# plt.show()


# đạo hàm và hàm mất mát
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w)-y)


def cost(w):
    N = Xbar.shape[0]
    return .5/N * np.linalg.norm(y - Xbar.dot(w), 2)**2

#  ---------------- IDEA: Công thức Taylor  ----------------
# https://mathworld.wolfram.com/TaylorSeries.html
#  ---------------- ----------------


# numerical gradient: công thức xấp xỉ hai phía ----------------
def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p)-cost(w_n))/(2*eps)
    return g


# Với các hàm số khác, cần viết lại hàm grad và cost ở trên
def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    # sai số nhỏ hơn nhỏ hơn 10^−6
    return True if np.linalg.norm(grad1-grad2) < 1e-6 else False


# print("Checking gradient: ", check_grad(np.random.rand(2, 1), cost, grad))
def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it)


w_init = np.array([[2], [1]])
w1, it1 = myGD(w_init, grad, 1)
# w1 is 2 dismension: đây mới là tham số tối ưu cho cái hàm đó
print(f'Solution myGD w = {w1[-1].T}, obtained after {it1} iterations')


# Đường đồng mức (level sets) ----------------
"""

# More reference:
# https://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent
