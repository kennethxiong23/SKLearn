# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import json
import numpy as np
# matplotlib 3.3.1
from matplotlib import pyplot
import cv2 as cv
import writeDataset
import argparse
import random

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--batch_size", required = False, default = 2,
                help = "batch size for where the split string sequence in data")
ap.add_argument("-r", "--random", required = False, default = False,
                help = "whether or not to use random data")
args = vars(ap.parse_args())

if args["random"]:
    writeDataset.randomDataset(int(args["batch_size"]), 'data.json')
else:
    writeDataset.createDataset(int(args["batch_size"]), "binaryData.txt", 'data.json')

readFile = open('data.json')
data = json.load(readFile)
pattern = np.array(data["pattern"])

digitsY = np.array(data["char"])
patternTrain, patternTest, endTrain, endTest = train_test_split(
    pattern, digitsY, test_size = 0.3, shuffle = True
    )
classifier = LogisticRegression(max_iter = 10000)
classifier.fit(patternTrain, endTrain)
preds = classifier.predict(patternTest)

correct = 0
incorrect = 0
for pred, gt in zip(preds, endTest):
    if pred == gt: correct += 1
    else: incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

plot_confusion_matrix(classifier, patternTest, endTest)
pyplot.show()

guess = input("give me a %s numbers " %args["batch_size"])

correct = 0
total = 0
while "exit" not in guess:
    pred = classifier.predict_proba([list(map(int,guess[-int(args["batch_size"]):-1]))])
    print("\nConfidence last is 0: %s\nConfidence last is 0: %s\n" %(pred[0][0], pred[0][1]))
    weight = random.random()
    if weight - pred[0][0] < 0:
        prediction = 0
    else:
        prediction = 1
    if str(prediction) == str(guess[-1]):
        correct += 1
        print("\033[92mCorrect end value was %s" %guess[-1])
        print("SKlearn correctly guessed %s\n\033[0m" %prediction)
    else:
        print("\033[91mCorrect end value was %s" %guess[-1])
        print("SKlearn incorrectly guessed %s\n\033[0m" %prediction)
    guess += input("Please enter another 1 or 0, enter exit to stop ")
    total += 1
printTuple = (correct, total, correct/total)
print("Logistic Regression correctly guessed %s of %s end values.\
 That's a correct rate of %s percent" %printTuple)
