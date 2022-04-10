import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

scores_dict = {
    "accuracy_score": accuracy_score,
    "f1_score": f1_score,
    "precision_score": precision_score,
    "recall_score": recall_score
}

def compute_metrics (y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, Y_pred = torch.max(y_pred_softmax, dim=1)
    metrics = {}
    values = np.unique(y_test)
    
    for i in values:
        metrics[i] = {}
        y_true = np.array((y_test == i)).astype("int")
        y_hat = np.array((Y_pred == i)).astype("int")

        for score_name, score_fn in scores_dict.items():
            metrics[i][score_name] = score_fn(y_true, y_hat)

    metrics = pd.DataFrame(metrics).T
    
    return metrics