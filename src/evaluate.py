# -*- coding: utf-8 -*-
"""实验评估：分类性能、定位性能、端到端任务指标、阈值分析、域偏移与失败案例。

产物：
    results/eval.json          全部指标
    results/fig_confusion.png  混淆矩阵
    results/fig_threshold.png  桥连类阈值—召回/精确率曲线
    results/fig_perclass.png   三种分类器的每类 F1
    results/fig_locate.png     两种定位器的召回与示例
    results/fig_domain.png     拍摄风格域偏移
    results/fig_failures.png   失败案例
"""

import csv
import json
import os
import pickle
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src import classify, features
from src.locate import ClassicLocator, grid_refine, load_locator, match_pairs, match_recall
from src.preprocess import crop_roi, gate, normalize, read_image
from src.route import decide

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RES = os.path.join(ROOT, 'results')
W = os.path.join(ROOT, 'weights')
CN = ['正常', '桥连', '虚焊少锡', '多锡', '偏移']
CN_SHORT = ['正常', '桥连', '少锡', '多锡', '偏移']


def load_rows(name='labels_phone.csv'):
    return list(csv.DictReader(open(os.path.join(DATA, name), encoding='utf-8')))


def split_of(bid, train_max=41, val_max=50):
    n = int(bid.split('_')[-1])
    if n <= train_max:
        return 'train'
    if n <= val_max:
        return 'val'
    return 'test'


def per_class(y_true, y_pred, n=5):
    out = []
    for c in range(n):
        tp = int(sum(1 for g, p in zip(y_true, y_pred) if g == c and p == c))
        fp = int(sum(1 for g, p in zip(y_true, y_pred) if g != c and p == c))
        fn = int(sum(1 for g, p in zip(y_true, y_pred) if g == c and p != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({'class': CN[c], 'support': tp + fn, 'tp': tp, 'fp': fp, 'fn': fn,
                    'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4)})
    return out


def macro_f1(per):
    return round(float(np.mean([p['f1'] for p in per])), 4)


def svm_probs(items):
    with open(os.path.join(W, 'svm.pkl'), 'rb') as f:
        model = pickle.load(f)
    X = np.vstack([features.extract(cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR),
                                                cv2.COLOR_BGR2GRAY)) for p, _ in items])
    return model.predict_proba(X)


def torch_probs(kind, items):
    model = (classify.load_resnet(os.path.join(W, 'resnet18.pt')) if kind == 'resnet18'
             else classify.load_shallow(os.path.join(W, 'shallow.pt')))
    return classify.predict_probs(model, [(p, 0) for p, _ in items])


def evaluate_classifiers():
    rows = load_rows()
    items = [(os.path.join(DATA, 'roi', r['roi_file']), int(r['label_id']))
             for r in rows if split_of(r['board_id']) == 'test']
    y = np.array([it[1] for it in items])
    out, probs = {}, {}

    res = {'svm': svm_probs, 'shallow': lambda it: torch_probs('shallow', it),
           'resnet18': lambda it: torch_probs('resnet18', it)}
    for k, fn in res.items():
        t0 = time.time()
        p = fn(items)
        probs[k] = p
        pred = p.argmax(1)
        per = per_class(y.tolist(), pred.tolist())
        out[k] = {'macro_f1': macro_f1(per), 'per_class': per,
                  'accuracy': round(float((pred == y).mean()), 4),
                  'seconds': round(time.time() - t0, 2)}
        print('[%s] test macro_f1=%.4f acc=%.4f' % (k, out[k]['macro_f1'], out[k]['accuracy']))
    return out, probs, y, items


def evaluate_locators():
    rows = load_rows()
    boards = sorted({r['board_id'] for r in rows if split_of(r['board_id']) == 'test'})
    truth = {b: [(float(r['cx']), float(r['cy'])) for r in rows if r['board_id'] == b]
             for b in boards}
    res = {}
    yolo_path = os.path.join(W, 'yolo_best.pt')
    locs = {'classic': ClassicLocator(), 'classic_grid': ClassicLocator()}
    if os.path.exists(yolo_path):
        try:
            locs['yolo'] = load_locator('yolo', yolo_path)[0]
        except Exception as e:
            print('yolo load failed:', e)

    for name, loc in locs.items():
        recs, t0 = [], time.time()
        for b in boards:
            img = normalize(read_image(os.path.join(DATA, 'raw', b + '.png')))
            bs = loc.locate(img)
            if name == 'classic_grid':
                bs = grid_refine(bs)
            rec, off, mm, tt = match_recall(bs, truth[b])
            recs.append((rec, off, len(bs), tt))
        res[name] = {
            'recall': round(float(np.mean([r[0] for r in recs])), 4),
            'avg_center_offset_px': round(float(np.mean([r[1] for r in recs])), 2),
            'avg_boxes': round(float(np.mean([r[2] for r in recs])), 2),
            'design_joints': int(np.mean([r[3] for r in recs])),
            'seconds_per_board': round((time.time() - t0) / len(boards), 3),
        }
        print('[locate:%s] recall=%.3f offset=%.1fpx boxes=%.1f/%.0f' % (
            name, res[name]['recall'], res[name]['avg_center_offset_px'],
            res[name]['avg_boxes'], res[name]['design_joints']))
    return res, boards, truth


def end_to_end(boards, truth, locator_kind='classic', clf_kind='resnet18', th=0.35,
               use_grid=True, rows=4, cols=6):
    loc, loc_name = load_locator(locator_kind, os.path.join(W, 'yolo_best.pt'))
    if clf_kind == 'resnet18':
        clf = classify.load_resnet(os.path.join(W, 'resnet18.pt'))
    elif clf_kind == 'shallow':
        clf = classify.load_shallow(os.path.join(W, 'shallow.pt'))
    else:
        with open(os.path.join(W, 'svm.pkl'), 'rb') as f:
            clf = pickle.load(f)

    stats = {'boards': 0, 'gate_reject': 0, 'truth_joints': 0, 'detected': 0,
             'defect_truth': 0, 'defect_hit': 0, 'review': 0, 'review_true_defect': 0,
             'auto': 0, 'time': []}
    rows = load_rows()
    for b in boards:
        img = read_image(os.path.join(DATA, 'raw', b + '.png'))
        t0 = time.time()
        ok, msg, _ = gate(img)
        if not ok:
            stats['gate_reject'] += 1
            continue
        img_n = normalize(img)
        boxes = loc.locate(img_n)
        if use_grid:
            boxes = grid_refine(boxes)
        rois = [crop_roi(img, (x[0] + x[2]) / 2, (x[1] + x[3]) / 2) for x in boxes]
        if not rois:
            probs = np.zeros((0, 5))
        elif clf_kind == 'svm':
            probs = clf.predict_proba(np.vstack(
                [features.extract(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)) for r in rois]))
        else:
            probs = classify.predict_probs(clf, [(r, 0) for r in rois])
        gts = [(float(r['cx']), float(r['cy']), int(r['label_id'])) for r in rows
               if r['board_id'] == b]
        stats['boards'] += 1
        stats['truth_joints'] += len(gts)
        for (tx, ty, g) in gts:
            if g != 0:
                stats['defect_truth'] += 1
        pairs = match_pairs(boxes, [(x, y) for (x, y, _) in gts])
        stats['detected'] += len(pairs)
        for ti, bi in pairs:
            g = gts[ti][2]
            pred = int(np.argmax(probs[bi]))
            d = decide(probs[bi], th=th)
            if d['action'] == 'review':
                stats['review'] += 1
                if g != 0:
                    stats['review_true_defect'] += 1
            else:
                stats['auto'] += 1
            if g != 0 and pred == g:
                stats['defect_hit'] += 1
        stats['time'].append(time.time() - t0)

    n_det = max(stats['detected'], 1)
    out = {
        'locator': loc_name, 'classifier': clf_kind, 'threshold': th,
        'boards': stats['boards'], 'gate_reject': stats['gate_reject'],
        'joint_recall': round(stats['detected'] / max(stats['truth_joints'], 1), 4),
        'defect_recall': round(stats['defect_hit'] / max(stats['defect_truth'], 1), 4),
        'review_rate': round(stats['review'] / n_det, 4),
        'review_hit_rate': round(stats['review_true_defect'] / max(stats['review'], 1), 4),
        'auto_rate': round(stats['auto'] / n_det, 4),
        'sec_per_board': round(float(np.mean(stats['time'])), 3),
    }
    print('[e2e:%s+%s] joint_recall=%.3f defect_recall=%.3f review=%.3f hit=%.3f %.2fs/board'
          % (loc_name, clf_kind, out['joint_recall'], out['defect_recall'],
             out['review_rate'], out['review_hit_rate'], out['sec_per_board']))
    return out


def threshold_curve(probs, y):
    """以桥连类概率为判据，扫描阈值，输出召回、精确率与复核率。"""
    pts = []
    pb = probs[:, 1]
    for th in np.arange(0.15, 0.96, 0.05):
        sel = pb >= th
        tp = int(((y == 1) & sel).sum())
        fp = int(((y != 1) & sel).sum())
        fn = int(((y == 1) & ~sel).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        pts.append((float(round(th, 2)), round(rec, 4), round(prec, 4), int((~sel).sum())))
    return pts


def domain_shift():
    """同一条模型在扫描风格（清晰、无退化）数据上的表现，用于量化域偏移。"""
    rows = load_rows('labels_scan.csv')
    items = [(os.path.join(DATA, 'roi', r['roi_file']), int(r['label_id'])) for r in rows]
    y = np.array([it[1] for it in items])
    out = {}
    for kind, fn in (('svm', svm_probs),
                     ('shallow', lambda it: torch_probs('shallow', it)),
                     ('resnet18', lambda it: torch_probs('resnet18', it))):
        p = fn(items)
        pred = p.argmax(1)
        per = per_class(y.tolist(), pred.tolist())
        out[kind] = {'macro_f1': macro_f1(per),
                     'accuracy': round(float((pred == y).mean()), 4), 'per_class': per}
        print('[domain:scan:%s] macro_f1=%.4f acc=%.4f' % (kind, out[kind]['macro_f1'],
                                                           out[kind]['accuracy']))
    return out


def plot_all(clf, probs, y, locs, domain):
    os.makedirs(RES, exist_ok=True)

    # 1 混淆矩阵（主模型）
    cm = np.zeros((5, 5), int)
    pred = probs['resnet18'].argmax(1)
    for g, p in zip(y, pred):
        cm[g, p] += 1
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=200)
    im = ax.imshow(cm, cmap='Blues')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() * 0.6 else 'black', fontsize=11)
    ax.set_xticks(range(5), CN_SHORT, fontsize=10)
    ax.set_yticks(range(5), CN_SHORT, fontsize=10)
    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    ax.set_title('ResNet18 迁移学习混淆矩阵（测试集 %d 个 ROI）' % len(y), fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_confusion.png'))
    plt.close(fig)

    # 2 每类 F1 对比
    labels = CN_SHORT
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    for k, (name, color) in enumerate([('svm', '#8c8c8c'), ('shallow', '#4d88c4'),
                                      ('resnet18', '#c0504d')]):
        vals = [c['f1'] for c in clf[name]['per_class']]
        ax.bar(x + (k - 1) * width, vals, width, label=name, color=color)
        for xi, v in zip(x + (k - 1) * width, vals):
            ax.text(xi, v + 0.02, '%.2f' % v, ha='center', fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('F1')
    ax.set_title('三种分类器的每类 F1（同一划分、同一测试集）', fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc='lower center')
    ax.grid(axis='y', ls=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_perclass.png'))
    plt.close(fig)

    # 3 阈值曲线
    pts = threshold_curve(probs['resnet18'], y)
    th = [p[0] for p in pts]
    rec = [p[1] for p in pts]
    prec = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.plot(th, rec, '-o', ms=4, color='#c0504d', label='桥连召回')
    ax.plot(th, prec, '-s', ms=4, color='#4d88c4', label='桥连精确率')
    for t in (0.30, 0.35, 0.40, 0.50):
        ax.axvline(t, ls='--', lw=0.8, color='#7f7f7f')
        ax.text(t, 0.05, '%.2f' % t, rotation=90, fontsize=8, ha='right', color='#555555')
    ax.set_xlabel('桥连类判定阈值')
    ax.set_ylabel('指标值')
    ax.set_ylim(0, 1.05)
    ax.set_title('桥连类阈值—召回/精确率（阈值越低越不容易漏检，但复核增多）', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(ls=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_threshold.png'))
    plt.close(fig)

    # 4 定位对比
    names = list(locs.keys())
    recalls = [locs[n]['recall'] for n in names]
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=200)
    bars = ax.bar(names, recalls, width=0.45,
                  color=['#8c8c8c', '#4d88c4'][:len(names)])
    for b, v in zip(bars, recalls):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, '%.3f' % v, ha='center', fontsize=10)
    ax.axhline(0.9, ls='--', color='#c0504d', lw=1)
    ax.text(0.03, 0.845, 'S2 目标 0.90', fontsize=9, color='#c0504d')
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('焊点定位召回（容差 22 px）')
    ax.set_title('两种定位路径的召回（测试 9 块板，共 216 个焊点）', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_locate.png'))
    plt.close(fig)

    # 5 域偏移：三种分类器在两种拍摄风格上的表现
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=200)
    keys = [('svm', '#8c8c8c'), ('shallow', '#4d88c4'), ('resnet18', '#c0504d')]
    x = np.arange(2)
    width = 0.26
    for k, (name, color) in enumerate(keys):
        vals = [clf[name]['macro_f1'], domain[name]['macro_f1']]
        ax.bar(x + (k - 1) * width, vals, width, label=name, color=color)
        for xi, v in zip(x + (k - 1) * width, vals):
            ax.text(xi, v + 0.02, '%.3f' % v, ha='center', fontsize=8)
    ax.set_xticks(x, ['手机拍摄（同分布）', '扫描风格（异分布）'])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('宏平均 F1')
    ax.set_title('域偏移：两种拍摄风格下的分类性能', fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc='lower center')
    ax.grid(axis='y', ls=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_domain.png'))
    plt.close(fig)


def failure_cases(probs, y, items, topn=6):
    """挑选置信度最高的错误样本，作为失败案例分析材料。"""
    pred = probs.argmax(1)
    conf = probs.max(1)
    errs = []
    for i, (g, p) in enumerate(zip(y, pred)):
        if g != p:
            errs.append((float(conf[i]), i, int(g), int(p)))
    errs.sort(reverse=True)
    picked = errs[:topn]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.2), dpi=200)
    for ax, (c, i, g, p) in zip(axes.ravel(), picked):
        img = cv2.cvtColor(cv2.imread(items[i][0]), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title('真值 %s / 预测 %s\n置信度 %.2f' % (CN[g], CN[p], c), fontsize=9)
        ax.axis('off')
    for ax in axes.ravel()[len(picked):]:
        ax.axis('off')
    fig.suptitle('失败案例（按置信度从高到低，均为错判）', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_failures.png'))
    plt.close(fig)
    return [{'roi': os.path.basename(items[i][0]), 'truth': CN[g], 'pred': CN[p],
             'conf': round(c, 4)} for (c, i, g, p) in picked]


def gate_sweep(boards):
    """对同一批板面逐级加模糊，量化质量门控在什么退化程度开始拒收。"""
    pts = []
    for sigma in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
        lap, rejected = [], 0
        for b in boards[:5]:
            img = read_image(os.path.join(DATA, 'raw', b + '.png'))
            if sigma > 0:
                img = cv2.GaussianBlur(img, (0, 0), sigma)
            ok, _, info = gate(img)
            lap.append(float(info['laplacian_var']))
            if not ok:
                rejected += 1
        pts.append({'sigma': sigma, 'mean_laplacian_var': round(float(np.mean(lap)), 1),
                    'reject_rate': round(rejected / 5.0, 3)})
        print('[gate] sigma=%.1f lap=%.1f reject=%.2f' % (sigma, np.mean(lap), rejected / 5.0))
    return pts


def oracle_end_to_end(boards, clf_kind='resnet18', th=0.35):
    """假设定位完全准确（直接用标注坐标裁剪），用于把端到端损失拆到定位与分类两段。"""
    if clf_kind == 'resnet18':
        clf = classify.load_resnet(os.path.join(W, 'resnet18.pt'))
    elif clf_kind == 'shallow':
        clf = classify.load_shallow(os.path.join(W, 'shallow.pt'))
    else:
        with open(os.path.join(W, 'svm.pkl'), 'rb') as f:
            clf = pickle.load(f)
    rows = load_rows()
    st = {'truth': 0, 'defect_truth': 0, 'defect_hit': 0, 'review': 0, 'review_true_defect': 0}
    for b in boards:
        img = read_image(os.path.join(DATA, 'raw', b + '.png'))
        gts = [(float(r['cx']), float(r['cy']), int(r['label_id'])) for r in rows
               if r['board_id'] == b]
        rois = [crop_roi(img, cx, cy) for (cx, cy, _) in gts]
        if clf_kind == 'svm':
            probs = clf.predict_proba(np.vstack(
                [features.extract(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)) for r in rois]))
        else:
            probs = classify.predict_probs(clf, [(r, 0) for r in rois])
        for i, (_, _, g) in enumerate(gts):
            st['truth'] += 1
            pred = int(np.argmax(probs[i]))
            if g != 0:
                st['defect_truth'] += 1
            d = decide(probs[i], th=th)
            if d['action'] == 'review':
                st['review'] += 1
                if g != 0:
                    st['review_true_defect'] += 1
            if g != 0 and pred == g:
                st['defect_hit'] += 1
    out = {'classifier': clf_kind,
           'defect_recall': round(st['defect_hit'] / max(st['defect_truth'], 1), 4),
           'review_rate': round(st['review'] / max(st['truth'], 1), 4),
           'review_hit_rate': round(st['review_true_defect'] / max(st['review'], 1), 4)}
    print('[oracle:%s] defect_recall=%.3f review=%.3f' % (clf_kind, out['defect_recall'],
                                                          out['review_rate']))
    return out


def plot_pipeline(clf, oracle, e2e):
    """把 ROI 级 F1、完美定位下的缺陷召回、真实端到端缺陷召回画成一条损失链。"""
    names = ['ROI 级宏 F1\n(标注裁剪)', '缺陷召回\n(定位完美)', '缺陷召回\n(完整流水线)']
    vals = [clf['resnet18']['macro_f1'], oracle['resnet18']['defect_recall'],
            e2e['classic_resnet18']['defect_recall']]
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
    bars = ax.bar(names, vals, width=0.5, color=['#4d88c4', '#8c8c8c', '#c0504d'])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, '%.3f' % v, ha='center', fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('指标值')
    ax.set_title('误差传导：从分类能力到端到端缺陷召回', fontsize=10)
    ax.grid(axis='y', ls=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'fig_pipeline.png'))
    plt.close(fig)


def main():
    os.makedirs(RES, exist_ok=True)
    clf, probs, y, items = evaluate_classifiers()
    locs, boards, truth = evaluate_locators()

    e2e = {}
    e2e['classic_svm'] = end_to_end(boards, truth, 'classic', 'svm', 0.35)
    e2e['classic_shallow'] = end_to_end(boards, truth, 'classic', 'shallow', 0.35)
    e2e['classic_resnet18'] = end_to_end(boards, truth, 'classic', 'resnet18', 0.35)
    e2e['classic_resnet18_th025'] = end_to_end(boards, truth, 'classic', 'resnet18', 0.25)
    e2e['classic_resnet18_th045'] = end_to_end(boards, truth, 'classic', 'resnet18', 0.45)
    # 消融：接入网格先验补全（提升定位召回，但会把假阳性拉成整列错位，最终未启用）
    e2e['classic_resnet18_grid'] = end_to_end(boards, truth, 'classic', 'resnet18', 0.35,
                                              use_grid=True)

    oracle = {'svm': oracle_end_to_end(boards, 'svm', 0.35),
              'resnet18': oracle_end_to_end(boards, 'resnet18', 0.35)}

    domain = domain_shift()
    gate_pts = gate_sweep(boards)
    failures = failure_cases(probs['resnet18'], y, items)
    plot_all(clf, probs, y, locs, domain)
    plot_pipeline(clf, oracle, e2e)

    thresholds = threshold_curve(probs['resnet18'], y)
    out = {'classifiers': clf, 'locators': locs, 'end_to_end': e2e, 'oracle': oracle,
           'threshold_curve': thresholds, 'domain_shift': domain,
           'gate_sweep': gate_pts,
           'failures': failures, 'test_size': int(len(y)),
           'test_boards': boards}
    json.dump(out, open(os.path.join(RES, 'eval.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print('saved -> results/eval.json')


if __name__ == '__main__':
    main()






