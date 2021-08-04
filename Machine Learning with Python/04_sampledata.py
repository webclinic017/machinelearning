import random
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def create_ds(hm, var, step=2, corr=False):
    '''
    hm: how many datapoints
    var: how much each point can vary from the previous point
    step: how far to step on average per point
    corr: data points correlation
    '''
    val = 1
    ys = []
    for i in range(hm):
        y = val+random.randrange(-var, var)
        ys.append(y)
        if corr and corr == 'pos':
            val += step
        elif corr and corr == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    # for i in range(len(ys)):
    #     print(xs[i], ys[i])
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit(xs, ys):
    m = (mean(xs)*mean(ys) - mean(xs*ys))/(mean(xs)**2-mean(xs**2))
    b = mean(ys) - m*mean(xs)
    return m, b


def sqr_err(ys_org, ys_line):
    return sum((ys_line-ys_org)**2)


def coeff_of_determination(ys_org, ys_line):
    y_mean_line = [mean(ys_org) for _ in ys_org]
    sqr_err_reg = sqr_err(ys_org, ys_line)
    sqr_err_y_mean = sqr_err(ys_org, y_mean_line)
    return 1-(sqr_err_reg/sqr_err_y_mean)


xs, ys = create_ds(40, 10, 2, corr='pos')
# xs, ys = create_ds(10, 10, 10, corr='pos')
# print(xs, ys)
m, b = best_fit(xs, ys)
regression_line = [(m*x)+b for x in xs]
r_squared = coeff_of_determination(ys, regression_line)
print(r_squared)


plt.scatter(xs, ys, color='b', label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
