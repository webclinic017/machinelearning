# Unsupervised learning: K-means clustering : Classification
# không biết nhãn, phân dữ liệu thành các cụm tính chất giống nhau
# cluster: tập hợp các điểm ở gần nhau trong một không gian nào đó

# một điểm đại diện (center) và những điểm xung quanh
# xét điểm gần center nào nhất thì nó thuộc về nhóm với center đó

# Hàm mất mát và bài toán tối ưu
# Thuật toán tối ưu hàm mất mát

# ------------------------------------------------
# maybe reference to Fibonacci Cluster
# ------------------------------------------------

""" # -----------------------
# ----------------- Segmentation Problem -----------------
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
# initialize the random number generator
np.random.seed(11)
''' # test
import random
random.seed()
print(random.random())

random.seed(10)
print(random.random())
'''
# tạo dữ liệu -----------------------
means = [[2, 2], [8, 3], [3, 6], [5, 9]]
# means = [[2, 2, 4], [8, 3, 6], [3, 6, 0], [5, 9, 1]]
# ma trận hiệp phương sai
cov = [[1, 0], [0, 1]]

# ValueError: mean and cov must have same length
# cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

N = 500
# toa do
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3 = np.random.multivariate_normal(means[3], cov, N)

# X0 = X0[:5]
# print(X0)
# print()
# # phan tu dau tien (toa do x)
# print(X0[:, 0])

# print(len(X0))

X = np.concatenate((X0, X1, X2, X3), axis=0)

# print(len(X))

K = 4

original_label = np.asarray([0]*N + [1]*N + [2]*N + [3]*N).T
# print(len(original_label))
# end tạo dữ liệu -----------------------


def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    X3 = X[label == 3, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)
    plt.plot(X3[:, 0], X3[:, 1], 'kD', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    # plt.show()


# kmeans_display(X, original_label)
def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers

    # number and size of each item: 2000 elems + vector (x,y)
    # print(X.shape)
    # Generates a random sample
    return X[np.random.choice(X.shape[0], k, replace=False)]


def kmeans_assign_labels(X, centers):
    # distance
    D = cdist(X, centers)
    # print(D)
    # return index of the closest
    return np.argmin(D, axis=1)


def kmeans_update_centers(X, labels, K):
    # shape[1] = 2 (for vector)
    centers = np.zeros((K, X.shape[1]))
    # 4 center - print 4 times
    # print(centers)

    for k in range(K):
        # collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]
        # print(Xk)

        # take average
        centers[k, :] = np.mean(Xk, axis=0)
    return centers


def has_converged(centers, new_centers):
    # compare two sets of centers
    return (set([tuple(a) for a in centers]) ==
            set([tuple(a) for a in new_centers]))


def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    # print(centers)
    labels = []
    # counting
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        # print(labels[-1])
        new_centers = kmeans_update_centers(X, labels[-1], K)
        # print(centers[-1])
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)


(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
# why -1 : last elem is most precious
print(centers[-1])
# print(centers)

kmeans_display(X, labels[-1])

# ----------------------- """

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# tạo dữ liệu -----------------------
means = [[2, 2], [8, 3], [3, 6], [5, 9]]
cov = [[1, 0], [0, 1]]
N = 500

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X3 = np.random.multivariate_normal(means[3], cov, N)

X = np.concatenate((X0, X1, X2, X3), axis=0)
# end tạo dữ liệu -----------------------


def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    X3 = X[label == 3, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)
    plt.plot(X3[:, 0], X3[:, 1], 'kD', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()


kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label)
