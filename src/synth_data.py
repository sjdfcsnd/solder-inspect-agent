# -*- coding: utf-8 -*-
"""程序化焊点图像生成器。

用途：在无法获取真实实训板面照片时，生成几何结构确定、缺陷类型可控、可完全复现的
焊点图像集，用于打通「采集—定位—分类—路由—复核」全链路，并支持退化强度可控的
域偏移实验。所有坐标与标签由生成过程记录，因此标注无人工误差。

用法：
    python -m src.synth_data --boards 60 --style phone --out data --seed 42
    python -m src.synth_data --boards 15 --style scan --out data --seed 7 --prefix scan
"""

import argparse
import csv
import os

import cv2
import numpy as np

CLASSES = ['normal', 'bridge', 'insufficient', 'excess', 'offset']
CLASS_CN = ['正常', '桥连', '虚焊少锡', '多锡', '偏移']

BOARD_W, BOARD_H = 1280, 720
GRID_C, GRID_R = 6, 4
SPACING_X, SPACING_Y = 178, 148
ORIGIN_X, ORIGIN_Y = 158, 168
ROI_HALF = 36
ROI_SIZE = 128

PCB_BGR = (58, 92, 44)
PAD_BGR = (38, 108, 168)
PIN_BGR = (196, 196, 198)
SILK_BGR = (238, 238, 238)


def _new_canvas(rng):
    img = np.full((BOARD_H, BOARD_W, 3), PCB_BGR, np.uint8)
    # 阻焊层的细微纹理
    tex = rng.normal(0, 6, (BOARD_H, BOARD_W, 1)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + tex, 0, 255).astype(np.uint8)
    # 板框丝印
    cv2.rectangle(img, (40, 36), (BOARD_W - 40, BOARD_H - 36), SILK_BGR, 2)
    cv2.rectangle(img, (52, 48), (BOARD_W - 52, BOARD_H - 48), SILK_BGR, 1)
    # 定位标记
    for (mx, my) in ((92, 92), (BOARD_W - 92, 92), (92, BOARD_H - 92)):
        cv2.circle(img, (mx, my), 13, (60, 60, 190), 2)
        cv2.circle(img, (mx, my), 5, (60, 60, 190), -1)
    return img


def _draw_pad_and_pin(img, cx, cy, rng):
    cv2.ellipse(img, (cx, cy), (28, 17), 0, 0, 360, PAD_BGR, -1)
    cv2.ellipse(img, (cx, cy), (28, 17), 0, 0, 360, (22, 70, 118), 2)
    w, h = 11, 46
    cv2.rectangle(img, (cx - w // 2, cy - h // 2 - 6), (cx + w // 2, cy + h // 2 - 6),
                  (168, 168, 170), -1)
    cv2.rectangle(img, (cx - w // 2, cy - h // 2 - 6), (cx + w // 2, cy + h // 2 - 6),
                  (120, 120, 122), 1)
    # 引脚端面
    cv2.line(img, (cx - w // 2, cy - h // 2 - 6), (cx + w // 2, cy - h // 2 - 6),
             (215, 215, 218), 2)


def _rc(rng, base, spread=12):
    return tuple(int(np.clip(base + rng.integers(-spread, spread + 1), 0, 255)) for _ in range(3))


def _draw_solder(img, cx, cy, cls, rng, target=None):
    """在焊盘上绘制锡堆。target 为桥连的目标点。

    各缺陷类的尺寸与亮度都带随机扰动，且与正常类的取值范围存在重叠，避免分类器
    只靠一个恒定几何特征就把类别分开。
    """
    if cls == 'normal':
        x0 = int(cx + rng.integers(-6, 7))
        y0 = int(cy - 3 + rng.integers(-5, 6))
        rx, ry = int(rng.integers(15, 30)), int(rng.integers(8, 17))
        cv2.ellipse(img, (x0, y0), (rx, ry), int(rng.integers(-10, 11)), 0, 360,
                    _rc(rng, 176, 22), -1)
        cv2.ellipse(img, (x0, y0), (rx, ry), 0, 0, 360, _rc(rng, 146, 16), 1)
        if rng.random() < 0.7:
            cv2.ellipse(img, (x0 + int(rng.integers(-rx // 2, 0)),
                              y0 + int(rng.integers(-ry // 2, 0))),
                        (int(rng.integers(3, 10)), int(rng.integers(2, 5))), 0, 0, 360,
                        _rc(rng, 226, 16), -1)
    elif cls == 'insufficient':
        x0 = int(cx + rng.integers(-6, 7))
        y0 = int(cy + rng.integers(0, 7))
        rx, ry = int(rng.integers(11, 21)), int(rng.integers(6, 13))
        cv2.ellipse(img, (x0, y0), (rx, ry), int(rng.integers(0, 60)), 0, 360,
                    _rc(rng, 118, 22), -1)
        cv2.ellipse(img, (x0, y0), (rx, ry), 0, 0, 360, _rc(rng, 92, 16), 1)
        if rng.random() < 0.55:
            cv2.ellipse(img, (x0 - int(rng.integers(4, 10)), y0 - 3),
                        (int(rng.integers(3, 7)), int(rng.integers(2, 7))), 25, 0, 360,
                        (34, 52, 34), -1)
    elif cls == 'excess':
        rx, ry = int(rng.integers(21, 35)), int(rng.integers(15, 27))
        cv2.ellipse(img, (cx, cy - 2), (rx, ry), 0, 0, 360, _rc(rng, 196, 16), -1)
        cv2.ellipse(img, (cx, cy - 2), (rx, ry), 0, 0, 360, _rc(rng, 160, 14), 1)
        cv2.ellipse(img, (cx - int(rng.integers(5, 13)), cy - int(rng.integers(6, 13))),
                    (int(rng.integers(7, 13)), int(rng.integers(4, 8))), 0, 0, 360,
                    _rc(rng, 234, 12), -1)
    elif cls == 'offset':
        dx = int(rng.choice([-1, 1]) * rng.integers(8, 19))
        dy = int(rng.choice([-1, 1]) * rng.integers(5, 13))
        rx, ry = int(rng.integers(16, 27)), int(rng.integers(9, 16))
        cv2.ellipse(img, (cx + dx, cy + dy - 3), (rx, ry), 0, 0, 360, _rc(rng, 178, 18), -1)
        cv2.ellipse(img, (cx + dx, cy + dy - 3), (rx, ry), 0, 0, 360, _rc(rng, 146, 16), 1)
        cv2.ellipse(img, (cx + dx - int(rng.integers(4, 9)), cy + dy - int(rng.integers(5, 11))),
                    (int(rng.integers(3, 9)), int(rng.integers(2, 5))), 0, 0, 360,
                    _rc(rng, 224, 14), -1)
    elif cls == 'bridge':
        rx, ry = int(rng.integers(17, 25)), int(rng.integers(10, 14))
        cv2.ellipse(img, (cx, cy - 3), (rx, ry), 0, 0, 360, _rc(rng, 184, 14), -1)
        if target is not None:
            tx, ty = target
            mx = (cx + tx) // 2 + int(rng.integers(-4, 5))
            my = (cy + ty) // 2 + int(rng.integers(-4, 5))
            length = float(np.hypot(tx - cx, ty - cy)) / 2.0 + 6
            ang = float(np.degrees(np.arctan2(ty - cy, tx - cx)))
            thick = int(rng.integers(4, 10))
            col = _rc(rng, 178, 16)
            cv2.ellipse(img, (mx, my), (int(length), thick), ang, 0, 360, col, -1)
            cv2.ellipse(img, (mx, my), (int(length), thick), ang, 0, 360, _rc(rng, 152, 12), 1)


def _specular(img, rng, n=2):
    for _ in range(n):
        h, w = img.shape[:2]
        cx = int(rng.integers(w // 7, w * 6 // 7))
        cy = int(rng.integers(h // 7, h * 6 // 7))
        rx = int(rng.integers(70, 210))
        ry = int(rng.integers(35, 95))
        ang = int(rng.integers(0, 180))
        layer = img.copy()
        cv2.ellipse(layer, (cx, cy), (rx, ry), ang, 0, 360, (252, 252, 254), -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 55)
        img = cv2.addWeighted(img, 0.82, layer, 0.18, 0)
    return img


def degrade(img, rng, style, M=None):
    """退化：返回 (img, M_used)。M 用于同步变换焊点坐标。"""
    if style == 'scan':
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        noise = rng.normal(0, 2.0, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8), None

    # 低分辨率细节丢失：先下采样再放大，模拟手机远距离拍摄与压缩
    s = float(rng.uniform(0.62, 0.85))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (BOARD_W, BOARD_H), interpolation=cv2.INTER_LINEAR)

    # 光照不均
    gx = np.linspace(rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), BOARD_W)
    gy = np.linspace(rng.uniform(-0.10, 0.10), rng.uniform(-0.10, 0.10), BOARD_H)
    grad = (1.0 + gy[:, None] + gx[None, :]).astype(np.float32)
    img = np.clip(img.astype(np.float32) * grad[:, :, None], 0, 255).astype(np.uint8)

    ang = float(rng.uniform(-2.4, 2.4))
    center = (BOARD_W / 2.0, BOARD_H / 2.0)
    M = cv2.getRotationMatrix2D(center, ang, float(rng.uniform(0.97, 1.03)))
    M[0, 2] += rng.uniform(-7, 7)
    M[1, 2] += rng.uniform(-6, 6)
    img = cv2.warpAffine(img, M, (BOARD_W, BOARD_H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    alpha = float(rng.uniform(0.74, 1.12))
    beta = float(rng.uniform(-24, 22))
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    img = _specular(img, rng, n=int(rng.integers(1, 4)))

    sigma = float(rng.uniform(2.0, 4.0))
    img = cv2.GaussianBlur(img, (0, 0), sigma)

    noise = rng.normal(0, float(rng.uniform(5, 9)), img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, M


def _apply_M(M, x, y):
    if M is None:
        return float(x), float(y)
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2],
            M[1, 0] * x + M[1, 1] * y + M[1, 2])


def build_board(board_id, rng, style):
    img = _new_canvas(rng)
    positions = []
    for r in range(GRID_R):
        for c in range(GRID_C):
            positions.append((ORIGIN_X + c * SPACING_X, ORIGIN_Y + r * SPACING_Y, r, c))

    labels = ['normal'] * len(positions)
    idx = list(range(len(positions)))
    plan = ['bridge', 'bridge', 'insufficient', 'insufficient',
            'excess', 'excess', 'offset', 'offset']
    chosen = rng.choice(idx, size=len(plan), replace=False)
    for pos_i, cls in zip(chosen, plan):
        labels[int(pos_i)] = cls

    for (x, y, r, c) in positions:
        _draw_pad_and_pin(img, x, y, rng)

    records = []
    for i, (x, y, r, c) in enumerate(positions):
        cls = labels[i]
        target = None
        if cls == 'bridge':
            if c < GRID_C - 1:
                target = (x + SPACING_X, y)
            elif r < GRID_R - 1:
                target = (x, y + SPACING_Y)
            else:
                target = (x - SPACING_X, y)
        _draw_solder(img, x, y, cls, rng, target)
        records.append((i, float(x), float(y), cls))

    img, M = degrade(img, rng, style)
    mapped = []
    for (i, x, y, cls) in records:
        nx, ny = _apply_M(M, x, y)
        mapped.append((i, nx, ny, cls))
    return img, mapped


def crop_roi(img, cx, cy, half=ROI_HALF, size=ROI_SIZE):
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - img.shape[1])
    pad_b = max(0, y1 - img.shape[0])
    if any([pad_l, pad_t, pad_r, pad_b]):
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t
    patch = img[y0:y1, x0:x1]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', type=int, default=60)
    ap.add_argument('--style', default='phone', choices=['phone', 'scan'])
    ap.add_argument('--out', default='data')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prefix', default='board')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'roi'), exist_ok=True)

    rows = []
    for b in range(args.boards):
        bid = '%s_%03d' % (args.prefix, b)
        img, recs = build_board(b, rng, args.style)
        raw_path = os.path.join(args.out, 'raw', bid + '.png')
        cv2.imwrite(raw_path, img)
        for (i, x, y, cls) in recs:
            roi = crop_roi(img, x, y)
            roi_name = '%s_r%02d.png' % (bid, i)
            cv2.imwrite(os.path.join(args.out, 'roi', roi_name), roi)
            rows.append([bid, i, round(x, 1), round(y, 1), cls, CLASSES.index(cls),
                         args.style, roi_name])

    label_path = os.path.join(args.out, 'labels_%s.csv' % args.style)
    with open(label_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['board_id', 'roi_index', 'cx', 'cy', 'label', 'label_id',
                    'style', 'roi_file'])
        w.writerows(rows)

    dist = {c: 0 for c in CLASSES}
    for r in rows:
        dist[r[4]] += 1
    print('boards=%d rois=%d' % (args.boards, len(rows)))
    print('labels ->', label_path)
    print('dist', dist)


if __name__ == '__main__':
    main()
