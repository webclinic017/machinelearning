'''# Classification: KNN, SVM
- group / clustering
- prior-labeled data for training
'''
import random
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def create_ds(hm, var, step=2, corr=False):
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


xs, ys = create_ds(10, 10, 2, corr='pos')
plt.scatter(xs, ys, color='b', label='data')
plt.legend(loc=4)
plt.show()
