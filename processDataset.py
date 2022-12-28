import pandas as pd

training_set = pd.read_csv('./archive/mitbih_train.csv', header=None)
test_set = pd.read_csv('./archive/mitbih_test.csv', header=None)

trainingSet = training_set.iloc[:, :187]
testSet = test_set.iloc[:, :187]
trainingLabels = training_set.iloc[:, 187]
testLabels = test_set.iloc[:, 187]
