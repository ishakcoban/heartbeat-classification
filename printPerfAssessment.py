def printMetrics(classifier, confusionMatrix, accuracy, precision, recall, f1_score,processingTime):
    print("-----------------" + classifier + "-----------------")
    print("Confusion matrix:\n", confusionMatrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1_score)
    print("--- %s seconds ---" % processingTime)
