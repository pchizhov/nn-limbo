import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    true_positives = np.sum((prediction == 1) & (ground_truth == 1))

    precision = true_positives / np.sum(prediction == 1)
    recall = true_positives / np.sum(ground_truth == 1)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = np.sum(prediction == ground_truth) / prediction.shape[0]

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    accuracy = np.sum(prediction == ground_truth) / prediction.shape[0]
    return accuracy
