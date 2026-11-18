# -*- coding: utf-8 -*-
"""焊点区域定位。

两条路径：
1) ClassicLocator：HSV 铜色/银色双通道分割 + 形态学闭运算 + 面积与长宽比过滤。
   焊盘边缘是铜色，即使锡量不足也能被分割出来，因此对虚焊、少锡的召回优于纯灰度
   Otsu 方案。无需训练、无需下载权重，作为主路径与降级路径。
2) YoloLocator：Ultralytics YOLOv8n 单类检测器，整板输入输出焊点外接框。
   本项目曾从零训练该检测器（12 轮，mAP50 为 0，见 docs/run_log.md），受预训练权重
   下载受限与样本量不足影响未采用；接口保留，权重存在时可切换。

定位之后可用 grid_refine 按板面 4×6 网格先验补全漏检框。
"""

import cv2
import numpy as np

BOX = 66
# 框中心与焊点中心之间的固定偏移：在验证集 9 块板、182 个焊点上估计，
# 中位偏移 dx=-0.5px、dy=+5.25px（标准差约 1.1px），说明偏差是分割掩码边界造成的
# 系统性下垂，而非随机误差，因此用常量平移校正，不做逐图拟合。
SHIFT_DX = -0.5
SHIFT_DY = 5.25


class ClassicLocator:
    name = 'classic'

    def __init__(self, min_area=350, max_area=4000, box=BOX, close_ksize=9, close_iter=2):
        self.min_area = min_area
        self.max_area = max_area
        self.box = box
        self.close_ksize = close_ksize
        self.close_iter = close_iter

    def locate(self, img, calibrate=True):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        copper = cv2.inRange(hsv, (6, 55, 50), (34, 255, 255))
        silver = cv2.inRange(hsv, (0, 0, 135), (180, 55, 255))
        mask = cv2.bitwise_or(copper, silver)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_ksize, self.close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=self.close_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not (0.60 <= w / float(h) <= 1.70):
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            if calibrate:
                cx += SHIFT_DX
                cy += SHIFT_DY
            boxes.append([cx - self.box / 2, cy - self.box / 2,
                          cx + self.box / 2, cy + self.box / 2, 0.5])
        return self.sort(boxes)

    @staticmethod
    def sort(boxes):
        boxes.sort(key=lambda b: (round((b[1] + b[3]) / 2 / 40), (b[0] + b[2]) / 2))
        return boxes


class YoloLocator:
    name = 'yolo'

    def __init__(self, weights, imgsz=640, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf

    def locate(self, img):
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, verbose=False)
        boxes = []
        for r in res:
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                boxes.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], float(b.conf[0])])
        return ClassicLocator.sort(boxes)


def load_locator(kind='auto', weights=None, imgsz=640, conf=0.25):
    """按配置加载定位器；auto 在权重缺失或加载失败时回退传统方法。"""
    import os
    if kind in ('yolo', 'auto') and weights and os.path.exists(weights):
        try:
            return YoloLocator(weights, imgsz=imgsz, conf=conf), 'yolo'
        except Exception as e:
            return ClassicLocator(), 'classic_fallback(%s)' % type(e).__name__
    return ClassicLocator(), 'classic'


def grid_refine(boxes, n_rows=4, n_cols=6, box=BOX):
    """利用板面焊点的规则网格先验补全漏检。

    电子实训板面按行列均匀排布焊点：先在纵、横方向各做一次一维聚类估计行列位置，
    再把缺失的网交点补成候选框；已有检测框保留其真实中心，不整体平移。
    这样可在几乎不增加误检的前提下把定位召回补到接近设计焊点数。
    """
    if len(boxes) < max(n_rows, n_cols) or n_rows * n_cols > 200:
        return boxes
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return boxes
    cx = np.array([(b[0] + b[2]) / 2.0 for b in boxes], dtype=float)
    cy = np.array([(b[1] + b[3]) / 2.0 for b in boxes], dtype=float)

    # 先剔除远离群体的离群框（多为板边反光或丝印造成的假阳性），避免把聚类中心拉偏
    med_x, med_y = np.median(cx), np.median(cy)
    mad_x = np.median(np.abs(cx - med_x)) + 1e-6
    mad_y = np.median(np.abs(cy - med_y)) + 1e-6
    keep = (np.abs(cx - med_x) < 3.5 * 1.4826 * mad_x) & (np.abs(cy - med_y) < 3.5 * 1.4826 * mad_y)
    if keep.sum() < max(n_rows, n_cols):
        return boxes
    cx, cy = cx[keep], cy[keep]
    kept = [b for b, k in zip(boxes, keep) if k]

    def centers(v, k):
        k = min(k, len(np.unique(np.round(v, 1))))
        if k < 2:
            return np.array([v.mean()])
        km = KMeans(n_clusters=k, n_init=2, random_state=0).fit(v.reshape(-1, 1))
        return np.sort(km.cluster_centers_.ravel())

    xs = centers(cx, n_cols)
    ys = centers(cy, n_rows)
    step = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
    tol = 0.42 * step
    out = []
    for y in ys:
        for x in xs:
            d = np.hypot(cx - x, cy - y)
            i = int(np.argmin(d))
            if d[i] <= tol:
                out.append(kept[i])
            else:
                out.append([x - box / 2, y - box / 2, x + box / 2, y + box / 2, 0.5])
    return ClassicLocator.sort(out)


def match_pairs(pred_boxes, truth_centers, tol=22.0):
    """真值与检测框的全局最优一对一匹配（匈牙利算法），返回 (真值索引, 框索引) 列表。

    逐点贪心匹配会让先遍历的真值抢走唯一的近邻框，进而把"漏检"错记成"错检"，
    因此这里按代价矩阵做一次全局最优指派，只保留距离在容差内的配对。
    """
    if not truth_centers or not pred_boxes:
        return []
    D = np.zeros((len(truth_centers), len(pred_boxes)))
    for i, (tx, ty) in enumerate(truth_centers):
        for j, b in enumerate(pred_boxes):
            D[i, j] = np.hypot((b[0] + b[2]) / 2 - tx, (b[1] + b[3]) / 2 - ty)
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(D)
    except Exception:
        ri, ci = [], []
        used = set()
        for i in range(len(truth_centers)):
            j = int(np.argmin(D[i]))
            if j not in used:
                used.add(j)
                ri.append(i)
                ci.append(j)
    return [(int(i), int(j)) for i, j in zip(ri, ci) if D[i, j] <= tol]


def match_recall(pred_boxes, truth_centers, tol=22.0):
    """以真值坐标为中心的容差匹配，返回定位召回与平均中心偏移。"""
    if not truth_centers:
        return 0.0, 0.0, 0, 0
    pairs = match_pairs(pred_boxes, truth_centers, tol)
    if not pairs:
        return 0.0, 0.0, 0, len(truth_centers)
    offs = [float(np.hypot((pred_boxes[j][0] + pred_boxes[j][2]) / 2 - truth_centers[i][0],
                           (pred_boxes[j][1] + pred_boxes[j][3]) / 2 - truth_centers[i][1]))
            for i, j in pairs]
    return len(pairs) / float(len(truth_centers)), float(np.mean(offs)), len(pairs), len(truth_centers)


