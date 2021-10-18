# From https://raw.githubusercontent.com/chrischoy/DeepGlobalRegistration/master/core/knn.py
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
#import torch
import numpy as np
from scipy.spatial import cKDTree


def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def match_feats(feat_src, feat_dst, mutual_filter=True, k=1):
    if not mutual_filter:
        nns01 = find_knn_cpu(feat_src, feat_dst, knn=1, return_distance=False)
        corres01_idx0 = np.arange(len(nns01)).squeeze()
        corres01_idx1 = nns01.squeeze()
        return np.stack((corres01_idx0, corres01_idx1)).T
    else:
        # for each feat in src, find its k=1 nearest neighbours
        nns01 = find_knn_cpu(feat_src, feat_dst, knn=1, return_distance=False)
        # for each feat in dst, find its k nearest neighbours
        nns10 = find_knn_cpu(feat_dst, feat_src, knn=k, return_distance=False)
        # find corrs
        num_feats = len(nns01)
        corres01 = []
        if k == 1:
            for i in range(num_feats):
                if i == nns10[nns01[i]]:
                    corres01.append([i, nns01[i]])
        else:
            for i in range(num_feats):
                if i in nns10[nns01[i]]:
                    corres01.append([i, nns01[i]])
        print(
            f'Before mutual filter: {num_feats}, after mutual_filter with k={k}: {len(corres01)}.'
        )

        # Fallback if mutual filter is too aggressive
        if len(corres01) < 10:
            nns01 = find_knn_cpu(feat_src,
                                 feat_dst,
                                 knn=1,
                                 return_distance=False)
            corres01_idx0 = np.arange(len(nns01)).squeeze()
            corres01_idx1 = nns01.squeeze()
            return np.stack((corres01_idx0, corres01_idx1)).T

        return np.asarray(corres01)
