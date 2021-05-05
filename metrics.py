# -*- coding: utf-8 -*-
from sklearn.metrics import f1_score
import torch

def calc_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.size()[1] == 1:
        return (predictions.round() == labels).sum().item() / labels.size(0)
    else:
        return (predictions.max(dim=1)[1] == labels.max(dim=1)[1]).sum().item() / labels.size(0)

def calc_f1score(predictions: torch.Tensor, labels: torch.Tensor, average: str = 'micro') -> float:
    """
    Calculate f1 score based on averaging method defined.

    Args:
        predictions: tensor with predictions
        labels: tensor with original labels
        average: averaging method

    Returns:
        f1 score
    """
    if predictions.size(1) > 1:
        y_pred = predictions.max(dim=1)[1].detach().cpu().numpy()
    else:
        y_pred = torch.reshape(predictions, (-1,)).round().detach().cpu().numpy()
    
    if labels.size(1) > 1:
        y_true = labels.max(dim=1)[1].detach().cpu().numpy()
    else:
        y_true = torch.reshape(labels, (-1,)).detach().cpu().numpy()
    
    return f1_score(y_true, y_pred, average=average)