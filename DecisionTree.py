from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from processDataset import trainingSet, testSet, trainingLabels, testLabels
from printPerfAssessment import printMetrics
import time


def implementDecisionTree():
    start_time = time.time()
    clf = DecisionTreeClassifier()

    clf = clf.fit(trainingSet, trainingLabels)

    y_pred = clf.predict(testSet)

    printMetrics("Decision Tree", metrics.confusion_matrix(testLabels, y_pred),
                 metrics.accuracy_score(testLabels, y_pred),
                 metrics.precision_score(testLabels, y_pred, average='macro'),
                 metrics.recall_score(testLabels, y_pred, average='macro'),
                 metrics.f1_score(testLabels, y_pred, average='macro'),
                 time.time() - start_time)