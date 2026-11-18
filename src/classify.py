# -*- coding: utf-8 -*-
"""缺陷分类：浅层卷积网络与 ResNet18 迁移学习。

三种路线对同一批 ROI 训练，共用同一份按板划分的划分文件，保证结果可比：
- svm      ：HOG+LBP + PCA + RBF 核支持向量机（见 train.py）
- shallow  ：三个卷积块，从头训练，用来隔离「预训练」带来的增益
- resnet18 ：ImageNet 预训练骨干，先冻结后微调
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

CLASSES = ['normal', 'bridge', 'insufficient', 'excess', 'offset']
CLASS_CN = {0: '正常', 1: '桥连', 2: '虚焊少锡', 3: '多锡', 4: '偏移'}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class RoiDataset(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        src, label = self.items[i]
        if isinstance(src, str):
            img = cv2.imread(src, cv2.IMREAD_COLOR)
        else:
            img = src
        if img is None:
            raise RuntimeError('ROI 读取失败：%s' % src)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img), torch.tensor(label, dtype=torch.long)


class ShallowCNN(nn.Module):
    def __init__(self, n_cls=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.25), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(n_cls=5, pretrained=True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_cls)
    return model


def freeze_backbone(model, freeze=True):
    for name, p in model.named_parameters():
        if not name.startswith('fc.'):
            p.requires_grad = not freeze


def train_torch(model, train_items, val_items, epochs=12, lr=1e-3, batch=32,
                class_weights=None, freeze_epochs=0, seed=42, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cpu'
    model.to(device)
    tr_loader = DataLoader(RoiDataset(train_items, train_tf), batch_size=batch,
                           shuffle=True, num_workers=0)
    va_loader = DataLoader(RoiDataset(val_items, eval_tf), batch_size=batch,
                           shuffle=False, num_workers=0)
    w = None
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    best_f1, best_state, history = -1.0, None, []

    cur_lr = lr
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cur_lr, weight_decay=1e-4)

    for ep in range(epochs):
        if freeze_epochs and ep == freeze_epochs:
            freeze_backbone(model, False)
            cur_lr = lr / 10.0
            opt = torch.optim.AdamW(model.parameters(), lr=cur_lr, weight_decay=1e-4)
            if verbose:
                print('  -> unfreeze at epoch %d, lr=%.1e' % (ep, cur_lr))

        model.train()
        tl, correct, total = 0.0, 0, 0
        for xb, yb in tr_loader:
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                preds.extend(model(xb).argmax(1).tolist())
                gts.extend(yb.tolist())
        f1 = macro_f1(gts, preds, len(CLASSES))
        history.append({'epoch': ep + 1, 'train_loss': tl / max(total, 1),
                        'train_acc': correct / max(total, 1), 'val_macro_f1': f1})
        if verbose:
            print('  ep %02d loss %.4f train_acc %.3f val_f1 %.3f' % (
                ep + 1, tl / max(total, 1), correct / max(total, 1), f1))
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1, history


def macro_f1(gts, preds, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = sum(1 for g, p in zip(gts, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(gts, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(gts, preds) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def predict_probs(model, items, batch=32):
    model.eval()
    loader = DataLoader(RoiDataset(items, eval_tf), batch_size=batch, shuffle=False)
    outs = []
    with torch.no_grad():
        for xb, _ in loader:
            outs.append(torch.softmax(model(xb), dim=1).numpy())
    return np.concatenate(outs, axis=0)


def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_shallow(path):
    m = ShallowCNN()
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


def load_resnet(path):
    m = build_resnet18(pretrained=False)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m
