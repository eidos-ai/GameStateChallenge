from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np

def clustering_accuracy(labels_true, labels_pred):
    # Create the confusion matrix
    cm = confusion_matrix(labels_true, labels_pred)
    
    # Find the optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    
    # Calculate the overall accuracy using the optimal assignment
    overall_accuracy = cm[row_ind, col_ind].sum() / cm.sum()

    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i, (row, col) in enumerate(zip(row_ind, col_ind)):
        class_accuracy = cm[row, col] / np.sum(cm[row, :])
        per_class_accuracy[row] = class_accuracy

    return overall_accuracy, per_class_accuracy

# Example usages:
# Pred <-> GT
# 0 <-> 2
# 1 <-> 0
# 2 <-> 1

# 1.0 accuracy
labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]  # Ground truth labels
labels_pred = [2, 2, 2, 0, 0, 0, 1, 1, 1, 1]  # Labels assigned by clustering algorithm
accuracy, per_class_accuracy = clustering_accuracy(labels_true, labels_pred)
print("Accuracy of clustering:", accuracy)
print("Accuracy per class:", per_class_accuracy)
print("==========")


# 0.7 accuracy
labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]  # Ground truth labels
labels_pred = [2, 2, 1, 0, 0, 2, 1, 1, 1, 0]  # Labels assigned by clustering algorithm
accuracy, per_class_accuracy = clustering_accuracy(labels_true, labels_pred)
print("Accuracy of clustering:", accuracy)
print("Accuracy per class:", per_class_accuracy)
print("==========")

# 0.6 accuracy
labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]  # Ground truth labels
labels_pred = [2, 2, 1, 0, 1, 2, 1, 1, 1, 0]  # Labels assigned by clustering algorithm
accuracy, per_class_accuracy = clustering_accuracy(labels_true, labels_pred)
print("Accuracy of clustering:", accuracy)
print("Accuracy per class:", per_class_accuracy)
print("==========")

# 0.5 accuracy
labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]  # Ground truth labels
labels_pred = [2, 0, 1, 0, 1, 2, 1, 1, 1, 0]  # Labels assigned by clustering algorithm
accuracy, per_class_accuracy = clustering_accuracy(labels_true, labels_pred)
print("Accuracy of clustering:", accuracy)
print("Accuracy per class:", per_class_accuracy)
print("==========")