# -*- coding: utf-8 -*-
"""训练三条路线（SVM / 浅层 CNN / ResNet18 迁移），并按板划分落盘划分文件。

划分规则：board_000—041 训练、042—050 验证、051—059 测试。同一块板的所有 ROI
只进入一个子集，避免同板相似外观造成测试集泄漏。
"""

import argparse
import csv
import glob
import json
import os
import pickle
import time

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import classify, features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RES = os.path.join(ROOT, 'results')
W = os.path.join(ROOT, 'weights')


def board_split(bid):
    n = int(bid.split('_')[-1])
    if n <= 41:
        return 'train'
    if n <= 50:
        return 'val'
    return 'test'


def load_items():
    rows = list(csv.DictReader(open(os.path.join(DATA, 'labels_phone.csv'), encoding='utf-8')))
    out = {'train': [], 'val': [], 'test': []}
    meta = []
    for r in rows:
        path = os.path.join(DATA, 'roi', r['roi_file'])
        item = (path, int(r['label_id']), r['board_id'])
        out[board_split(r['board_id'])].append(item)
        meta.append([r['board_id'], r['roi_file'], int(r['label_id']), board_split(r['board_id'])])
    with open(os.path.join(DATA, 'splits', 'splits.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['board_id', 'roi_file', 'label_id', 'split'])
        wr.writerows(meta)
    return out


def class_weights(items, n_cls=5):
    cnt = np.zeros(n_cls)
    for _, y, _ in items:
        cnt[y] += 1
    cnt[cnt == 0] = 1.0
    w = (cnt.sum() / (n_cls * cnt))
    # 桥连漏检代价最高，额外加权
    w[1] *= 1.5
    return w.tolist()


def run_svm(sp):
    t0 = time.time()
    print('[svm] 提取 HOG+LBP 特征 ...')
    Xtr = np.vstack([features.extract(cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR),
                                                    cv2.COLOR_BGR2GRAY)) for p, _, _ in sp['train']])
    ytr = np.array([y for _, y, _ in sp['train']])
    Xva = np.vstack([features.extract(cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR),
                                                    cv2.COLOR_BGR2GRAY)) for p, _, _ in sp['val']])
    yva = np.array([y for _, y, _ in sp['val']])

    pipe = Pipeline([('sc', StandardScaler()),
                     ('pca', PCA(n_components=0.95, random_state=42)),
                     ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True,
                                 random_state=42))])
    grid = GridSearchCV(pipe, {'svm__C': [0.5, 2, 8], 'svm__gamma': ['scale', 0.01]},
                        cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    yp = best.predict(Xva)
    f1 = f1_score(yva, yp, average='macro')
    print('[svm] best=%s  val_macro_f1=%.4f  dim=%d  %.1fs' % (
        grid.best_params_, f1, best.named_steps['pca'].n_components_, time.time() - t0))
    os.makedirs(W, exist_ok=True)
    with open(os.path.join(W, 'svm.pkl'), 'wb') as f:
        pickle.dump(best, f)
    return {'model': 'svm', 'val_macro_f1': float(f1), 'best_params': grid.best_params_,
            'pca_dim': int(best.named_steps['pca'].n_components_),
            'seconds': round(time.time() - t0, 1)}


def run_torch(kind, sp, epochs):
    t0 = time.time()
    tr = [(p, y) for p, y, _ in sp['train']]
    va = [(p, y) for p, y, _ in sp['val']]
    w = class_weights(sp['train'])
    if kind == 'shallow':
        model = classify.ShallowCNN()
        model, f1, hist = classify.train_torch(model, tr, va, epochs=epochs, lr=5e-4,
                                               class_weights=w)
        classify.save(model, os.path.join(W, 'shallow.pt'))
    else:
        model = classify.build_resnet18()
        model, f1, hist = classify.train_torch(model, tr, va, epochs=epochs, lr=1e-3,
                                               class_weights=w, freeze_epochs=4)
        classify.save(model, os.path.join(W, 'resnet18.pt'))
    print('[%s] val_macro_f1=%.4f  %.1fs' % (kind, f1, time.time() - t0))
    return {'model': kind, 'val_macro_f1': float(f1), 'epochs': epochs,
            'class_weights': [round(x, 4) for x in w],
            'seconds': round(time.time() - t0, 1), 'history': hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='all', choices=['all', 'svm', 'shallow', 'resnet18'])
    ap.add_argument('--epochs', type=int, default=12)
    args = ap.parse_args()

    sp = load_items()
    print('train=%d val=%d test=%d' % (len(sp['train']), len(sp['val']), len(sp['test'])))
    metrics = {}
    path = os.path.join(RES, 'train_metrics.json')
    if os.path.exists(path):
        metrics = json.load(open(path, encoding='utf-8'))

    if args.only in ('all', 'svm'):
        metrics['svm'] = run_svm(sp)
    if args.only in ('all', 'shallow'):
        metrics['shallow'] = run_torch('shallow', sp, args.epochs)
    if args.only in ('all', 'resnet18'):
        metrics['resnet18'] = run_torch('resnet18', sp, args.epochs)

    os.makedirs(RES, exist_ok=True)
    json.dump(metrics, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('saved ->', path)


if __name__ == '__main__':
    main()
