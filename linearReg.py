# python 3.7
# Scikit-learn ver. 0.23.2
import sklearn.linear_model
import sklearn.datasets
import sklearn.preprocessing

digits = sklearn.datasets.load_digits()
digitsX = digits.images
digitsX = digitsX.reshape((len(digitsX), 64))
digitsY = digits.target
