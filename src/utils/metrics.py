import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torchmetrics import IoU,JaccardIndex
import torch
import torch.nn.functional as F
import torch.nn as nn

def calculate_f1score(pred,gt):
    f1 = f1_score(gt, pred, average=None, zero_division = 1)
    return f1.mean()


def calculate_precision(pred,gt):
    score = precision_score(gt, pred, average=None, zero_division = 1)
    return score.mean()


def calculate_recall(pred,gt):
    score = recall_score(gt, pred, average=None, zero_division = 1)
    return score.mean()
  

   

#######https://github.com/sunxm2357/AdaShare/blob/master/utils/util.py
def get_iou(pred, gt, n_classes):
    
    total_miou = 0.0

    for pred_tmp, gt_tmp in zip(pred,gt):
        # pred_tmp = pred[i,:,:]
        # gt_tmp = gt[i,:,:]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):

            match = (pred_tmp == j)*1 + (gt_tmp == j)*1           


            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        miou = (sum(iou) / len(iou))
        total_miou += miou
        # print('total_miou:', total_miou)


    total_miou = total_miou / len(pred)
    return total_miou



def sn_metrics(pred, gt):
    pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    labels = gt.max(dim=1)[0] != 1

    gt = gt[labels]
    pred = pred[labels]

    gt = F.normalize(gt.float(), dim=1)
    pred = F.normalize(pred, dim=1)
    cosine_similiarity = nn.CosineSimilarity()
    cos_similarity = cosine_similiarity(gt, pred).detach().cpu().numpy()

    overall_cos = np.clip(cos_similarity, -1, 1)
    all_angles = np.arccos(overall_cos) / np.pi * 180.0  



    return overall_cos.mean(), all_angles


def edge_metrics(pred, gt):
    binary_mask = (gt != 1)
    edge_output_true = pred.masked_select(binary_mask)
    edge_gt_true = gt.masked_select(binary_mask)
    abs_err = torch.abs(edge_output_true - edge_gt_true)    
    abs_err = abs_err.mean()
    return abs_err.detach().cpu().numpy()



