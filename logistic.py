# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
digitsX = digits.images.reshape(len(digits.images), 64)
digitsY = digits.target
trainX, testX, trainY, testY = train_test_split(
    digitsX, digitsY, test_size = 0.3, shuffle = True
    )

classifier = LogisticRegression()
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)
