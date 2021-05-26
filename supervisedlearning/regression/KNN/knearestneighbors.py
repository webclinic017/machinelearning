# Classification and Regression.

# K-nearest neighbor : K number of neighbor -----------------
# tìm đầu ra của một điểm dữ liệu mới bằng cách chỉ dựa
# trên thông tin của K điểm dữ liệu trong training set gần
#  nó nhất (K-lân cận), không quan tâm
# đến việc có một vài điểm dữ liệu trong những điểm gần nhất này là nhiễu

# Khoảng cách trong không gian vector -----------------

# KNN phải nhớ tất cả các điểm dữ liệu training
# bất lợi bộ nhớ và thời gian tính toán

# -----------------
# Câu hỏi	---- Điểm dữ liệu ---- Data point
# Đáp án ----	Đầu ra, nhãn ---- Output, Label
# Ôn thi ---- Huấn luyện ---- Training
# Tập tài liệu mang vào phòng thi----Tập dữ liệu tập huấn----Training set
# Đề thi----Tập dữ liểu kiểm thử----Test set
# Câu hỏi trong dề thi----Dữ liệu kiểm thử----Test data point
# Câu hỏi có đáp án sai----Nhiễu----Noise, Outlier
# Câu hỏi gần giống----Điểm dữ liệu gần nhất----Nearest Neighbor
# -----------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_result(default_case=0):
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target

    # ba loại hoa Iris, Mỗi loại có 50 bông hoa
    print(f"Number of classes: {len(np.unique(iris_y))}")
    print(f"Number of data points: {len(iris_y)}")

    # tách 150 dữ liệu ra thành 2 phần: training set và test set
    # print(iris_y)
    X0 = iris_X[iris_y == 0, :]
    # print('\nSamples from class 0:\n', X0[:5, :])

    X1 = iris_X[iris_y == 1, :]
    # print('\nSamples from class 1:\n', X1[:5, :])

    X2 = iris_X[iris_y == 2, :]
    # print('\nSamples from class 2:\n', X2[:5, :])

    X_train, X_test, y_train, y_test = train_test_split(
        iris_X, iris_y, test_size=50)

    # print(f"Training size: {len(y_train)}")
    # print(f"Test size: {len(y_test)}")

    # xét 1 điểm training data gần nhất và lấy label của điểm
    #  đó để dự đoán cho điểm test này.

    # -----------------------------------------------
    if default_case == 0:
        # p=2 for euclidean_distance, KNN classifier with k=1 (n_neighbors)
        clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
        clf.fit(X_train, y_train)
        # predict class label
        y_pred = clf.predict(X_test)
        print(f"Accuracy of 1NN: {100*accuracy_score(y_test, y_pred):.2f}")

    # print("Print results for 20 test data points:")
    # print(f"Predicted labels: {y_pred[20:40]}")
    # # ground truth chính là nhãn/label/đầu ra thực sự của các điểm
    # # trong test data
    # print(f"Ground truth    : {y_test[20:40]}")

    # -----------------------------------------------
    elif default_case == 1:
        # tăng độ chính xác là tăng số lượng điểm lân cận lên, ví dụ 10 điểm
        # trong 10 điểm gần nhất, class nào chiếm đa số thì dự đoán kết quả
        #  là class (loai hoa) đó.
        clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
        clf.fit(X_train, y_train)
        # predict class label
        y_pred = clf.predict(X_test)
        print(
            f"Accuracy of 10NN major voting: {100*accuracy_score(y_test, y_pred):.2f}")

    # --------------- IDEAS FOR TRADING SYSTEM: trọng số cao ----------
    elif default_case == 2:
        # Trong kỹ thuật major voting bên trên, mỗi trong 10 điểm gần nhất
        # có vai trò như nhau và giá trị lá phiếu của mỗi điểm này là như nhau

        # những điểm gần hơn nên có trọng số cao hơn: đơn giản nhất là lấy
        # nghịch đảo của khoảng cách : mặc định của weights là 'uniform'
        clf = neighbors.KNeighborsClassifier(
            n_neighbors=10, p=2, weights='distance')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(
            f"Accuracy of 10NN (weight): {100*accuracy_score(y_test, y_pred):.2f}")
    elif default_case == 3:
        clf = neighbors.KNeighborsClassifier(
            n_neighbors=10, p=2, weights=myweight)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(
            f"Accuracy of 10NN (customized weight): {100*accuracy_score(y_test, y_pred):.2f}")
    else:
        print("Wrong param")
    # -----------------------------------------------


def myweight(distances):
    # we can change this number
    sigma2 = .5
    return np.exp(-distances**2/sigma2)


# get_result(3)

# ----------------- KNN cho Regression
# ước lượng đầu ra dựa trên đầu ra và khoảng cách của các điểm trong K-lân cận
# ước lượng như thế nào phải tự định nghĩa

# ----------------- Chuẩn hóa dữ liệu:
# ko thể khoảng cách vừa là cm vừa là mm đc -> lệch khoảng cách
# -> phải đưa các thuộc tính khác đơn vị đo về cùng khoảng giá trị

# ----------------- Sử dụng các phép đo khoảng cách khác nhau:
# norm1 norm2 ...

# ----------------- Tăng tốc cho KNN: K-D Tree và Ball Tree.
# https://www.google.com/search?sxsrf=ALeKk01lGDgf12Fa_GHN8GoPExDU-oBvqg:1622003630517&q=kd+tree+search&spell=1&sa=X&ved=2ahUKEwi1oOKxwubwAhVFUd4KHY6tDrgQBSgAegQIARAw&biw=960&bih=915
# https://en.wikipedia.org/wiki/Ball_tree
