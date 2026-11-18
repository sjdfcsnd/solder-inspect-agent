# -*- coding: utf-8 -*-
"""图像预处理与质量门控。

门控的意义：拍摄条件不满足的输入不应进入模型。若把失焦、过曝的图片送进分类器，
模型会给出一个看似正常的高置信度错误结论，比直接返回「请重拍」危害更大。
"""

import cv2
import numpy as np

LAP_THRESHOLD = 60.0
OVEREXPOSE_RATIO = 0.20
DARK_RATIO = 0.15


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('无法解码图像: %s' % path)
    return img


def quality_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    over = float((gray >= 250).mean())
    dark = float((gray <= 12).mean())
    return {'laplacian_var': float(lap), 'overexposed': over, 'dark': dark,
            'mean_gray': float(gray.mean())}


def gate(img):
    """返回 (通过?, 原因, 指标)。"""
    m = quality_metrics(img)
    if m['laplacian_var'] < LAP_THRESHOLD:
        return False, '图像模糊（拉普拉斯方差 %.1f < %.1f），请稳定后重拍' % (
            m['laplacian_var'], LAP_THRESHOLD), m
    if m['overexposed'] > OVEREXPOSE_RATIO:
        return False, '局部过曝面积 %.0f%%，请避开强反光或改变入射角' % (m['overexposed'] * 100), m
    if m['dark'] > DARK_RATIO:
        return False, '图像过暗，请补光后重拍', m
    return True, 'ok', m


def normalize(img):
    """抑制反光、提升局部对比度，保持尺寸不变。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    # 高光抑制：对接近饱和的像素做软压缩
    out_f = out.astype(np.float32)
    hi = out_f > 240
    out_f[hi] = 240 + (out_f[hi] - 240) * 0.35
    return np.clip(out_f, 0, 255).astype(np.uint8)


def crop_roi(img, cx, cy, half=36, size=128):
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - img.shape[1]), max(0, y1 - img.shape[0])
    if any([pad_l, pad_t, pad_r, pad_b]):
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        x0, x1 = x0 + pad_l, x1 + pad_l
        y0, y1 = y0 + pad_t, y1 + pad_t
    patch = img[y0:y1, x0:x1]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
