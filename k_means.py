"""
Recognise hand-written digits by loading the data from the Python
package and creating k-means algorithm to cluster and analyse the data.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from numpy.linalg import norm
from sklearn.decomposition import PCA
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

# Randomly selecting the data
tsne = TSNE(random_state=17)

# Set 2D data-set
pca = PCA(n_components=2)

# Load the digits using the import.
# Store the data of the table into the dataset variable.
digits = load_digits()
dataset = digits.data

n_samples, n_features = digits.data.shape
# Show there are 1797 samples.
print("n_samples: ", n_samples)
# Show there are 64 features.
print("n_features: ", n_features)


class KMeans:
    """
    The KMeans class has the attribute n_clusters, max_iterator and
    random_state. This is my implementation of the k-means clustering
    algorithm.
    """

    def __init__(self, n_clusters, max_iterator=300, random_state=0):
        """
        A constructor for this class pass the n_clusters, max_iterator,
        random_state as arguments and initialising these values into
        initial variables.
        Set the old_centroids and labels are set as array list.

        :param n_clusters:                  number of cluster
        :param max_iterator:                integer
        :param random_state:                integer
        """
        self.n_clusters = n_clusters
        self.max_iterator = max_iterator
        self.random_state = random_state

        self.old_centroids = []
        self.labels = []

    def init_centroids(self, items):
        """
        This method initialise the centroids randomly with in the
        dataset scale.

        :param items:                       dataset
        :return:                            centroid
        """
        np.random.RandomState(self.random_state)
        random_poss = np.random.permutation(items.shape[0])
        centroids = items[random_poss[: self.n_clusters]]
        return centroids

    def compute_centroids(self, items, label):
        """
        This function compute the cluster centroids.

        :param items:                       dataset
        :param label:                       image label
        :return:                            cluster centroids
        """
        centroids = np.zeros((self.n_clusters, items.shape[1]))
        for j in range(self.n_clusters):
            centroids[j, :] = np.mean(items[label == j, :], axis=0)
        return centroids

    def calculate_distance(self, items, centroid):
        """
        Function which calculates the distance using the Euclidean
        method. This equation allows to get the sum of the squared
        distance between each data points and each centroids.

        :param items:                       dataset
        :param centroid:                    centroid
        :return:                            sum of the squared distance
        """
        distance = np.zeros((items.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            squared_norm = norm(items - centroid[k, :], axis=1)
            distance[:, k] = np.square(squared_norm)
        return distance

    def get_closest_cluster(self, distance):
        """
        Method finds the closest clusters.

        :param distance:                    sum of the squared distance
        :return:                            indices of the min distance
                                                in a particular axis
        """
        return np.argmin(distance, axis=1)

    def compute_sse(self, items, label, centroid):
        """
        This compute the SSE to get the sum of squared distance (SSE)
        between data points and their assigned clusters.

        :param items:                       dataset
        :param label:                       image label
        :param centroid:                    centroid
        :return:                            sum of the squared distance
        """
        distance = np.zeros(items.shape[0])
        for k in range(self.n_clusters):
            distance[label == k] = norm(items[label == k] - centroid[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, items):
        """
        Function to compute k-means clustering. This done by computing
        the centroids for the clusters by taking the average of the all
        data points that belong to each cluster.

        :param items:                       dataset
        :return:                            void
        """
        self.centroids = self.init_centroids(items)
        for i in range(self.max_iterator):
            self.old_centroids = self.centroids
            distance = self.calculate_distance(items, self.old_centroids)
            self.labels = self.get_closest_cluster(distance)
            self.centroids = self.compute_centroids(items, self.labels)
            if np.all(self.old_centroids == self.centroids):
                break
        self.error = self.compute_sse(items, self.labels, self.centroids)

    def predict_clusters(self, items):
        """
        After calculating the distance with the items and the
        old_centroid, it will return predict the closest
        cluster distance.

        :param items:                       dataset
        :return:                            cluster
        """
        distance = self.calculate_distance(items, self.old_centroids)
        return self.get_closest_cluster(distance)


# Shows that K-means created 10 clusters in 64 features.
km = KMeans(n_clusters=10)
km.fit(digits.data)
clusters = km.predict_clusters(digits.data)

# Get images showing clusters centroids learned by k-means.
# Display 10 clusters centroids images with 8 x 8 pixels.
figure, axis = plt.subplots(2, 5, figsize=(8, 8))
centers = km.centroids.reshape(10, 8, 8)
for ax, center in zip(axis.flat, centers):
    ax.set(xticks=[], yticks=[])
    ax.imshow(center, interpolation="nearest", cmap=plt.cm.binary)
plt.show()

# Matching the learned cluster labels with true label found in them.
# Check the accuracy by finding similar digits within the data.
labels = np.zeros_like(clusters)
for i in range(10):
    mask = clusters == i
    labels[mask] = mode(digits.target[mask])[0]
print("Accuracy: ", accuracy_score(digits.target, labels))

data_scaled = pca.fit_transform(digits.data)
# fit multiple k-means algorithms.
# Store the values in an empty list.
total_variation = []
for cluster in range(1, 20):
    km = KMeans(n_clusters=cluster)
    km.fit(data_scaled)

    # Sum of distances of samples to their closest cluster center
    total_variation.append(km.compute_sse(data_scaled, km.labels, km.centroids))

# Convert the results into a dataframe and plotting them.
# Set Elbow Method to give a good k number of clusters
# based on the sum of squared distance (SSE) between data points
# and assigned clusters’ centroids.
frame = pd.DataFrame({"Cluster": range(1, 20), "SSE": total_variation})
plt.figure(figsize=(12, 6))
plt.plot(frame["Cluster"], frame["SSE"], marker="o")
plt.title("Elbow Method to give an optimal k value", fontweight="bold")
plt.xlabel("Number of clusters")
plt.ylabel("Total Variation")
plt.show()

# Show total points in each cluster.
km = KMeans(n_clusters=10)
km.fit(digits.data)
pred = km.predict_clusters(digits.data)
frame = pd.DataFrame(digits.data)
frame["cluster"] = pred
frame["cluster"].value_counts()
print("Clusters VS total points:")
print(frame["cluster"].value_counts())

# Shows the scatter plot of the data colored
# by the cluster they belong to.
# Choosing K=4 and the symbol ‘*‘ is the centroid of each cluster.

# PCA graph method
X_std = pca.fit_transform(digits.data)
km = KMeans(n_clusters=10)
km.fit(X_std)
centroids = km.centroids
labels = km.labels

plt.title("PCA: Handwritten clustered digits using Kmeans", fontweight="bold")
plt.scatter(
    X_std[:, 0],
    X_std[:, 1],
    c=labels,
    edgecolor="none",
    alpha=0.7,
    s=40,
    cmap=plt.cm.get_cmap("nipy_spectral", 10),
)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="*",
    s=300,
    c="r",
)
plt.colorbar()
plt.show()

# TSNE graph method
X_tsne = tsne.fit_transform(digits.data)
km = KMeans(n_clusters=10)
km.fit(X_tsne)
labels = km.labels
centroids = km.centroids

plt.figure(figsize=(12, 10))
plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=labels,
    edgecolor="none",
    alpha=0.7,
    s=40,
    cmap=plt.cm.get_cmap("nipy_spectral", 15),
)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="*",
    s=300,
    c="r",
    cmap=plt.cm.get_cmap("nipy_spectral", 50),
)
plt.colorbar()
plt.title("TSNE: Handwritten clustered digits using Kmeans", fontweight="bold")
plt.show()
