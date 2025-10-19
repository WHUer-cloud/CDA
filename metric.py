import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn import metrics

def cos_dist(f1, f2):
    """1 - cosine similarity"""
    return 1 - torch.nn.functional.cosine_similarity(f1, f2, dim=1)

def anomaly_score(e_list, d_list, img_size=256):
    """
    e_list: [e1,e2,e3]  encoder feature
    d_list: [d1,d2,d3,
             l1m,l1n,l2m,l2n,l3m,l3n]  decoder feature
    """
    e1, e2, e3 = e_list
    d1, d2, d3, l1m, l1n, l2m, l2n, l3m, l3n = d_list
    a1 = cos_dist(e1, d1)
    a2 = cos_dist(e2, d2)
    a3 = cos_dist(e3, d3)
    ac1 = cos_dist(e1, l1n) + cos_dist(e1, l1m)
    ac2 = cos_dist(e2, l2n) + cos_dist(e2, l2m)
    ac3 = cos_dist(e3, l3n) + cos_dist(e3, l3m)
    # 插值
    a1 = torch.nn.functional.interpolate(a1.unsqueeze(1), img_size, mode='bilinear', align_corners=True)
    a2 = torch.nn.functional.interpolate(a2.unsqueeze(1), img_size, mode='bilinear', align_corners=True)
    a3 = torch.nn.functional.interpolate(a3.unsqueeze(1), img_size, mode='bilinear', align_corners=True)
    ac1 = torch.nn.functional.interpolate(ac1.unsqueeze(1), img_size, mode='bilinear', align_corners=True)
    ac2 = torch.nn.functional.interpolate(ac2.unsqueeze(1), img_size, mode='bilinear', align_corners=True)
    ac3 = torch.nn.functional.interpolate(ac3.unsqueeze(1), img_size, mode='bilinear', align_corners=True)
    # 加权
    score = a1 + a2 + a3 + 0,2* (0.4 * ac1 + 0.2 * ac2 + 0.4 * ac3)
    score = score.squeeze(1).cpu().numpy()
    # 高斯平滑
    for i in range(score.shape[0]):
        score[i] = gaussian_filter(score[i], sigma=4)
    return score  # ndarray, (B,H,W)

def compute_auroc(scores, labels, gts):
    """
    scores: list of ndarray (H,W)
    labels: list  0-good,1-anomaly
    gts:    list of ndarray (H,W) 0/1
    """
    scores = np.concatenate([s.ravel() for s in scores])
    gts = np.concatenate([g.ravel() for g in gts])
    img_scores = np.array([s.max() for s in scores.reshape(-1, 256*256)])
    auroc_img = metrics.roc_auc_score(labels, img_scores)
    auroc_pix = metrics.roc_auc_score(gts, scores)
    return round(auroc_img, 3), round(auroc_pix, 3)
