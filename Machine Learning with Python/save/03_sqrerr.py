# càng lớn càng tốt
# high variance -> low fit
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)


def best_fit(xs, ys):
    m = (mean(xs)*mean(ys) - mean(xs*ys))/(mean(xs)**2-mean(xs**2))
    b = mean(ys) - m*mean(xs)
    return m, b


def sqr_err(ys_org, ys_line):
    return sum((ys_line-ys_org)**2)


def coeff_of_determination(ys_org, ys_line):
    y_mean_line = [mean(ys_org) for _ in ys_org]
    # y_mean_line = [mean(ys_org)]*len(ys_org)
    # print(y_mean_line)
    sqr_err_reg = sqr_err(ys_org, ys_line)
    sqr_err_y_mean = sqr_err(ys_org, y_mean_line)
    return 1-(sqr_err_reg/sqr_err_y_mean)


m, b = best_fit(xs, ys)
regression_line = [(m*x)+b for x in xs]
predict_x = 8
predict_y = m*predict_x + b
r_squared = coeff_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys, color='b', label='data')
plt.scatter(predict_x, predict_y, color='g', label='predict')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
