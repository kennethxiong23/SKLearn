import json
import random

def createDataset(batchSize, readFile, writeFile):
    """
    Purpose: Creates a dataset for sklearn thats split data into a specificed batch
    size. The end character of that batch becomes the label.
    Parameters: batchSize(int), the readfile(string), the writeFile(string)
    Return Val: None
    """
    readFile = open(readFile, encoding='utf-8-sig')
    preprocessData = readFile.read().strip()
    readFile.close()
    finalData = {}
    patternList = []
    lastList = []
    for i in range(0, len(preprocessData)):
        try:
            pattern = []
            for char in preprocessData[i:i+batchSize-1]:
                pattern.append(int(char))
            patternList.append(pattern)
            lastList.append(int(preprocessData[i+batchSize]))
        except IndexError:
            patternList.pop(-1)
            break

    finalData["pattern"] = patternList
    finalData["char"] = lastList

    json_object = json.dumps(finalData, indent = 4)
    writeFile = open(writeFile , "w")
    writeFile.write(json_object)

def randomDataset(batchSize, writeFile):
    """
    Purpose: Creates a dataset for sklearn that splits data into a specificed batch
    size. The data in this case is randommly generated 1s and 0s. The end
    character of that batch becomes the label.
    Parameters: batchSize(int), the readfile(string), the writeFile(string)
    Return Val: None
    """
    preprocessData = ""
    for i in range(0,10000):
        guess = random.randrange(0,2)
        preprocessData += str(guess)
    finalData = {}
    patternList = []
    lastList = []
    for i in range(0, len(preprocessData)):
        try:
            pattern = []
            for char in preprocessData[i:i+batchSize-1]:
                pattern.append(int(char))
            patternList.append(pattern)
            lastList.append(int(preprocessData[i+batchSize]))
        except IndexError:
            patternList.pop(-1)
            break

    finalData["pattern"] = patternList
    finalData["char"] = lastList

    json_object = json.dumps(finalData, indent = 4)
    writeFile = open(writeFile , "w")
    writeFile.write(json_object)
if __name__ == "__main__":
    createDataset(2, "binaryData.txt", "data.json")
