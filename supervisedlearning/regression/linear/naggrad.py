# thuật toán GD: NAG ----------------
# Khi tới gần đích, momemtum vẫn mất khá nhiều thời gian trước khi
# dừng lại vì đà chưa hết

# NAG khắc phục điều này, giúp cho thuật toán hội tụ nhanh hơn

# Ý tưởng cơ bản là dự đoán hướng đi trong tương lai
# ko sử dụng grad của điểm hiện tại, sử dụng grad của điểm tiếp theo

# implement NAG --------------- ???
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim
from matplotlib.animation import FuncAnimation, PillowWriter

np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3*X + .2*np.random.randn(1000, 1)

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)

print(f'Solution found by formula: w = {w_lr.T}')


def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w)-y)


def cost(w):
    N = Xbar.shape[0]
    return .5/N * np.linalg.norm(y - Xbar.dot(w), 2)**2


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


def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1-grad2) < 1e-6 else False


def GD_NAG(w_init, grad, eta, gamma):
    w = [w_init]
    v = [np.zeros_like(w_init)]
    for it in range(100):
        v_new = gamma*v[-1] + eta*grad(w[-1] - gamma*v[-1])
        w_new = w[-1] - v_new
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
        v.append(v_new)
    return (w, it)


w_init = np.array([[2], [1]])
w_mm, it_mm = GD_NAG(w_init, grad, .5, 0.9)

N = X.shape[0]
a1 = np.linalg.norm(y, 2)**2/N
b1 = 2*np.sum(X)/N
c1 = np.linalg.norm(X, 2)**2/N
d1 = -2*np.sum(y)/N
e1 = -2*X.T.dot(y)/N

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
xg = np.arange(1.5, 7.0, delta)
yg = np.arange(0.5, 4.5, delta)
Xg, Yg = np.meshgrid(xg, yg)
Z = a1 + Xg**2 + b1*Xg*Yg + c1*Yg**2 + d1*Xg + e1*Yg


def save_gif2(eta, gamma):
    (w, it) = GD_NAG(w_init, grad, eta, gamma)
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.cla()
    plt.axis([1.5, 7, 0.5, 4.5])
#     x0 = np.linspace(0, 1, 2, endpoint=True)

    def update(ii):
        if ii == 0:
            plt.cla()
            CS = plt.contour(Xg, Yg, Z, 100)
            manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3)]
            animlist = plt.clabel(
                CS, inline=.1, fontsize=10, manual=manual_locations)
#             animlist = plt.title('labels at selected locations')
            plt.plot(w_lr[0], w_lr[1], 'go')
        else:
            animlist = plt.plot([w[ii-1][0], w[ii][0]],
                                [w[ii-1][1], w[ii][1]], 'r-')
        animlist = plt.plot(w[ii][0], w[ii][1], 'ro', markersize=4)
        xlabel = '$\eta =$ ' + str(eta) + '; iter = %d/%d' % (ii, it)
        xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(grad(w[ii]))
        ax.set_xlabel(xlabel)
        return animlist, ax

    anim1 = FuncAnimation(fig, update, frames=np.arange(0, it), interval=200)
    plt.show()
#     fn = 'img2_' + str(eta) + '.gif'
    # # for file saving purpose
    # fn = 'LR_NAG_contours.gif'
    # my_writer = PillowWriter(fps=20, codec='libx264', bitrate=2)
    # anim1.save(fn, writer=my_writer, dpi=100)


eta = 1
gamma = .9
save_gif2(eta, gamma)
