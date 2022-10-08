# K-Means Clustering

Recognise hand-written digits by loading the data from the Python
package as follows:

from sklearn.datasets import load digits
digits = load digits()

The dataset contain 1,797 samples and 64 features. I created own
kmeans algorithm to cluster and analyse the data by finding the
accuracy in the similarity of the digits within the data, using
Elbow method to give an optimal k value, using PCA graph method
and TSNE graph method to show the clustered data in the scatter
graph.

I created own kMeans class containing the functions are:
  initialise_centroids
  compute_centroids
  calculate_distance
  find_closest_cluster
  compute_SSE
  fit
  predict_likeliest_clusters

### Prerequisites and Install

What things you need to install the software and how to install
them

```
install python 3.6.0+
```
```
pip install scikit-learn
```
```
pip install matplotlib
```
```
pip install pandas
```
```
pip install numpy
```
```
pip install scipy
```

## Running the code

Running the main application
```
python k-means.py
```

## Built With
* Python

## License

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details
