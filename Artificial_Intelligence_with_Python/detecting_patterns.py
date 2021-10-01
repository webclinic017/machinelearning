import sys
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.cluster import (
    KMeans,
    MeanShift,
    estimate_bandwidth
)
from sklearn.metrics import silhouette_score


def clustering_data(num_clusters=5):
    # Load data
    X = np.loadtxt('data\\data_clustering.txt', delimiter=',')
    # Plot input data
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none',
                edgecolors='black', s=80)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.title('Input data')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    # plt.show()

    # create KMeans object
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)

    # Train the KMeans clustering model
    kmeans.fit(X)

    # Step size of the mesh
    step_size = 0.01

    # Define the grid of points to plot the boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                                 np.arange(y_min, y_max, step_size))

    # Predict output labels for all the points on the grid
    output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Plot different regions and color them
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.clf()
    plt.imshow(output, interpolation='nearest',
               extent=(x_vals.min(), x_vals.max(),
                       y_vals.min(), y_vals.max()),
               cmap=plt.cm.Paired,
               aspect='auto',
               origin='lower')
    # Overlay input points
    plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none',
                edgecolors='black', s=80)
    # Plot the centers of clusters
    cluster_centers = kmeans.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                marker='o', s=210, linewidths=4, color='black',
                zorder=12, facecolors='black')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.title('Boundaries of clusters')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def mean_shift():
    # Load data
    X = np.loadtxt('data\\data_clustering.txt', delimiter=',')

    # Estimate the bandwidth of X
    bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))
    # print(bandwidth_X)

    # Cluster data with MeanShift
    mean_shift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
    mean_shift_model.fit(X)

    # Extract the centers of clusters
    cluster_centers = mean_shift_model.cluster_centers_
    print('\nCenters of clusters:\n', cluster_centers)

    # Estimate the number of clusters
    labels = mean_shift_model.labels_
    num_clusters = len(np.unique(labels))
    print("\nNumber of clusters in input data =", num_clusters)

    # Plot the points and cluster centers
    plt.figure()
    markers = 'o*xvs'
    for i, marker in zip(range(num_clusters), markers):
        # Plot points that belong to the current cluster
        plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker,
                    color='black')
        # Plot the cluster center
        cluster_center = cluster_centers[i]
        plt.plot(cluster_center[0], cluster_center[1], marker='o',
                 markerfacecolor='black', markeredgecolor='black',
                 markersize=15)
    plt.title('Clusters')
    plt.show()


def silhouette_scoring():
    # Load data
    X = np.loadtxt('data\\data_quality.txt', delimiter=',')

    # Initialize variables
    scores = []
    values = np.arange(2, 10)
    print(values)

    # Iterate through the defined range
    for num_clusters in values:
        # Train the KMeans clustering model
        kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
        kmeans.fit(X)
        # Estimate the silhouette score using Euclidean distance
        score = silhouette_score(
            X, kmeans.labels_, metric='euclidean', sample_size=len(X))
        print("\nNumber of clusters =", num_clusters)
        print("Silhouette score =", score)
        scores.append(score)
    # Plot silhouette scores
    plt.figure()
    plt.bar(values, scores, width=0.7, color='black', align='center')
    plt.title('Silhouette score vs number of clusters')

    # Extract best score and optimal number of clusters
    num_clusters = np.argmax(scores) + values[0]
    print('\nOptimal number of clusters =', num_clusters)

    # Plot data
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], color='black', s=80, marker='o',
                facecolors='none')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plt.title('Input data')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def gmms():
    return


def main():
    # clustering_data()
    # mean_shift()
    silhouette_scoring()


if __name__ == "__main__":
    main()
