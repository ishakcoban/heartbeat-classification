from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from processDataset import trainingSet, testSet, trainingLabels, testLabels
from printPerfAssessment import printMetrics
import time


def implementkNN():
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=11)

    knn.fit(trainingSet, trainingLabels)
    y_pred = knn.predict(testSet)

    printMetrics("kNN", metrics.confusion_matrix(testLabels, y_pred),
                 metrics.accuracy_score(testLabels, y_pred),
                 metrics.precision_score(testLabels, y_pred, average='macro'),
                 metrics.recall_score(testLabels, y_pred, average='macro'),
                 metrics.f1_score(testLabels, y_pred, average='macro'),
                 time.time() - start_time)
