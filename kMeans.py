# python 3.7
# Scikit-learn ver. 0.23.2
import sklearn.cluster
import sklearn.datasets
import sklearn.preprocessing

digits = sklearn.datasets.load_digits()
digitsX = digits.images
digitsX = digitsX.reshape((len(digitsX), 64))
digitsY = digits.target

estimator = sklearn.cluster.KMeans(
    init="k-means++",
    n_clusters = 10,
    n_init = 10,
)
estimator.fit(sklearn.preprocessing.scale(digitsX))
