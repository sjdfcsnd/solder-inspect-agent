# -*- coding: utf-8 -*-
"""传统特征：HOG + LBP，用于支持向量机基线。

这两类特征都是课程内的经典手工特征：HOG 描述梯度方向分布，对形状与轮廓敏感；
LBP 描述局部纹理，对整体亮度变化相对稳健。两者拼接后降维，作为「无学习表示」
的对照基线。HOG 按 cell/block 的标准定义用 NumPy 实现，避免依赖具体版本的
OpenCV 绑定。
"""

import cv2
import numpy as np


def hog_feature(gray, cell=16, bins=9, block=2):
    g = cv2.resize(gray, (128, 128)).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins
    nc = 128 // cell
    hist = np.zeros((nc, nc, bins), np.float32)
    for i in range(nc):
        for j in range(nc):
            m = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            a = ang[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            idx = np.floor(a).astype(np.int32) % bins
            np.add.at(hist[i, j], idx, m)
    # 2x2 block 归一化
    feats = []
    for i in range(nc - block + 1):
        for j in range(nc - block + 1):
            blk = hist[i:i + block, j:j + block, :].ravel()
            n = np.sqrt((blk ** 2).sum() + 1e-6)
            feats.append(blk / n)
    return np.concatenate(feats)


def lbp_hist(gray, radius=1):
    g = cv2.resize(gray, (128, 128)).astype(np.int16)
    code = np.zeros_like(g, dtype=np.uint32)
    shifts = [(-radius, 0), (radius, 0), (0, -radius), (0, radius),
              (-radius, -radius), (-radius, radius), (radius, -radius), (radius, radius)]
    for k, (dy, dx) in enumerate(shifts):
        nb = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        code |= ((nb >= g).astype(np.uint32) << k)
    hist, _ = np.histogram(code, bins=np.arange(0, 257), range=(0, 256))
    hist = hist.astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def extract(gray):
    return np.concatenate([hog_feature(gray), lbp_hist(gray)])
