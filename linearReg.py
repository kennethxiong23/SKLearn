# python 3.7
# Scikit-learn ver. 0.23.2
import sklearn.linear_model
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection

digits = sklearn.datasets.load_digits()
digitsX = digits.images
digitsX = digitsX.reshape((len(digitsX), 64))
digitsY = digits.target

trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(
    digitsX, digitsY, test_size = 0.2, shuffle = True
)

regModel = sklearn.linear_model.LinearRegression()
regModel.fit(trainX, trainY)

print(regModel.coef_)
