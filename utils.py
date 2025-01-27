import numpy as np
import random
import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import EvalPrediction


def seed_everything(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_log(log, path):
    with open(path, 'a') as f:
        f.writelines(log + '\n')


def multi_class_metrics(predictions, labels):
    softmax = torch.nn.Softmax()
    probs = softmax(torch.Tensor(predictions))
    y_pred = np.argmax(probs, axis=-1)
    y_true = labels
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    w_f1 = f1_score(y_true, y_pred, average='weighted')
    ma_f1 = f1_score(y_true, y_pred, average="macro")
    mi_f1 = f1_score(y_true, y_pred, average="micro")
    # return as dictionary
    metrics = {'accuracy': accuracy, 'weighted-f1': w_f1, 'macro-f1': ma_f1, 'micro-f1': mi_f1}
    labels_set = set(np.unique(labels).tolist())
    for _id in labels_set:
        metrics[str(_id)] = f1[_id]
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_class_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result
