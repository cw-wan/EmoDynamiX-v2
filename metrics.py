import numpy as np


def preference_bias(confusion_matrix, iteration=20):
    eps = 1e-6
    num_classes = confusion_matrix.shape[0]
    p = [1. for _ in range(num_classes)]

    # Update p iteratively
    for _ in range(iteration):
        for i in range(num_classes):
            n = sum([confusion_matrix[j, i] * p[j] / (p[i] + p[j] + eps) for j in range(num_classes)])
            d = sum([confusion_matrix[i, j] / (p[i] + p[j] + eps) for j in range(num_classes)])
            p[i] = n / d
        # Normalize
        p = [p[i] * num_classes / sum(p) for i in range(num_classes)]

    # Computing bias
    bias = (sum([(p[i] - np.mean(p)) ** 2 for i in range(num_classes)]) / num_classes) ** 0.5

    return bias
