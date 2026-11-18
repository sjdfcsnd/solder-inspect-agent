# -*- coding: utf-8 -*-
"""Agent 编排：把门控、定位、分类、路由串成一次完整运行。

命令行：
    python -m src.agent --image data/raw/board_052.png --save results/demo_board052.png
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from src import classify
from src.locate import grid_refine, load_locator
from src.preprocess import crop_roi, gate, normalize, read_image
from src.route import CLASS_CN, decide

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXPECTED = 24


class ClassifierWrapper:
    """统一三种分类器的概率输出接口。"""

    def __init__(self, kind='resnet18', weights_dir=None):
        self.kind = kind
        wd = weights_dir or os.path.join(ROOT, 'weights')
        if kind == 'resnet18':
            self.model = classify.load_resnet(os.path.join(wd, 'resnet18.pt'))
        elif kind == 'shallow':
            self.model = classify.load_shallow(os.path.join(wd, 'shallow.pt'))
        elif kind == 'svm':
            import pickle
            from src import features
            with open(os.path.join(wd, 'svm.pkl'), 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise ValueError('unknown classifier: %s' % kind)

    def probs(self, rois_bgr):
        if self.kind == 'svm':
            from src import features
            X = np.vstack([features.extract(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY))
                           for r in rois_bgr])
            return self.model.predict_proba(X)
        batch = torch.stack([classify.eval_tf(cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
                             for r in rois_bgr])
        with torch.no_grad():
            return torch.softmax(self.model(batch), dim=1).numpy()


class SolderAgent:
    def __init__(self, locator='auto', classifier='resnet18', th=0.35,
                 expected_joints=DEFAULT_EXPECTED, yolo_weights=None, use_grid=True):
        w = yolo_weights or os.path.join(ROOT, 'weights', 'yolo_best.pt')
        self.locator, self.locator_name = load_locator(locator, w, imgsz=640, conf=0.25)
        self.clf = ClassifierWrapper(classifier)
        self.classifier_name = classifier
        self.th = th
        self.expected = expected_joints
        self.use_grid = use_grid

    def run(self, image_path, save_dir=None, tag=''):
        t0 = time.time()
        img = read_image(image_path)
        ok, msg, qm = gate(img)
        base = {'image': os.path.basename(image_path), 'locator': self.locator_name,
                'classifier': self.classifier_name, 'threshold': self.th,
                'quality': qm}
        if not ok:
            base.update({'status': 'needs_retake', 'message': msg, 'joints': [],
                         'summary': {'n_joints': 0, 'n_defect': 0, 'n_review': 0,
                                     'review_rate': 0.0},
                         'elapsed_s': round(time.time() - t0, 3)})
            return base

        img_n = normalize(img)
        boxes = self.locator.locate(img_n)
        # 板面焊点按 4×6 规则排布：用网格先验补全漏检，再把补出的框标为低置信度
        if self.use_grid and self.expected == 24:
            boxes = grid_refine(boxes, n_rows=4, n_cols=6)
        if self.expected and abs(len(boxes) - self.expected) / float(self.expected) > 0.4:
            base.update({'status': 'incomplete', 'joints': [],
                         'message': '检出焊点 %d 个，与板面设计 %d 个偏差过大，请重拍或补拍'
                                    % (len(boxes), self.expected),
                         'summary': {'n_joints': len(boxes), 'n_defect': 0, 'n_review': 0,
                                     'review_rate': 0.0},
                         'elapsed_s': round(time.time() - t0, 3)})
            return base

        rois = []
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            rois.append(crop_roi(img_n, cx, cy))
        probs = self.clf.probs(rois) if rois else np.zeros((0, 5))

        joints = []
        for i, (b, p) in enumerate(zip(boxes, probs)):
            d = decide(p, th=self.th)
            joints.append({
                'index': i + 1,
                'bbox': [round(float(b[0]), 1), round(float(b[1]), 1),
                         round(float(b[2]), 1), round(float(b[3]), 1)],
                'label': CLASS_CN[d['label']], 'label_id': d['label'],
                'conf': round(d['conf'], 4), 'action': d['action'], 'reason': d['reason'],
                'probs': [round(float(x), 4) for x in p],
            })

        n_def = sum(1 for j in joints if j['label_id'] != 0)
        n_rev = sum(1 for j in joints if j['action'] == 'review')
        elapsed = time.time() - t0
        base.update({
            'status': 'ok',
            'joints': joints,
            'summary': {'n_joints': len(joints), 'n_defect': n_def, 'n_review': n_rev,
                        'review_rate': round(n_rev / max(len(joints), 1), 3),
                        'auto_rate': round(1 - n_rev / max(len(joints), 1), 3)},
            'message': '完成：%d 个焊点，%d 个缺陷，%d 个待复核' % (len(joints), n_def, n_rev),
            'elapsed_s': round(elapsed, 3),
        })
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            p = os.path.join(save_dir, 'annotated_%s.png' % (tag or
                              os.path.splitext(os.path.basename(image_path))[0]))
            save_annotated(img, base, p)
            base['annotated'] = p
        return base


def save_annotated(img, result, path):
    from PIL import Image, ImageDraw, ImageFont
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    if not os.path.exists(font_path):
        font_path = 'C:/Windows/Fonts/simhei.ttf'
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    f = ImageFont.truetype(font_path, 17)
    fs = ImageFont.truetype(font_path, 20)

    s = result['summary']
    head = '焊点 %d  缺陷 %d  待复核 %d (%.0f%%)' % (
        s['n_joints'], s['n_defect'], s['n_review'], s['review_rate'] * 100)
    dr.rectangle([0, 0, im.width, 34], fill=(24, 24, 24))
    dr.text((12, 6), head, font=fs, fill=(255, 255, 255))

    for j in result['joints']:
        x1, y1, x2, y2 = j['bbox']
        if j['action'] == 'review':
            color = (255, 196, 0)
        elif j['label_id'] != 0:
            color = (235, 60, 60)
        else:
            color = (60, 200, 90)
        dr.rectangle([x1, y1, x2, y2], outline=color, width=2)
        txt = '%s %.2f' % (j['label'], j['conf'])
        dr.text((x1 - 2, max(y1 - 22, 36)), txt, font=f, fill=color)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--locator', default='auto', choices=['auto', 'yolo', 'classic'])
    ap.add_argument('--classifier', default='resnet18',
                    choices=['resnet18', 'shallow', 'svm'])
    ap.add_argument('--th', type=float, default=0.35)
    ap.add_argument('--expected', type=int, default=DEFAULT_EXPECTED)
    ap.add_argument('--save', default=None, help='结果图与 json 的输出目录')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    agent = SolderAgent(args.locator, args.classifier, args.th, args.expected)
    res = agent.run(args.image, save_dir=args.save)
    print(json.dumps({k: res[k] for k in ('status', 'message', 'summary', 'elapsed_s',
                                          'locator', 'classifier')},
                     ensure_ascii=False, indent=2))
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        json.dump(res, open(args.json_out, 'w', encoding='utf-8'), ensure_ascii=False,
                  indent=2)


if __name__ == '__main__':
    main()




