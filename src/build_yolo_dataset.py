# -*- coding: utf-8 -*-
"""把合成真值转换为 YOLO 单类检测数据集（类别 0 = solder joint）。

按板划分：训练 0—41、验证 42—50、测试 51—59，保证同一块板不出现在两个子集中。
"""

import csv
import os
import shutil

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
YOLO = os.path.join(DATA, 'yolo')
BOX_W, BOX_H = 66, 70


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, 'labels_phone.csv'), encoding='utf-8')))
    by_board = {}
    for r in rows:
        by_board.setdefault(r['board_id'], []).append((float(r['cx']), float(r['cy'])))

    for sub in ('train', 'val', 'test'):
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(YOLO, d, sub), exist_ok=True)

    def split_of(bid):
        n = int(bid.split('_')[-1])
        if n <= 41:
            return 'train'
        if n <= 50:
            return 'val'
        return 'test'

    for bid, pts in sorted(by_board.items()):
        sub = split_of(bid)
        src = os.path.join(DATA, 'raw', bid + '.png')
        img = cv2.imread(src)
        h, w = img.shape[:2]
        dst_img = os.path.join(YOLO, 'images', sub, bid + '.jpg')
        if not os.path.exists(dst_img):
            cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        for (cx, cy) in pts:
            lines.append('0 %.6f %.6f %.6f %.6f' % (
                cx / w, cy / h, BOX_W / w, BOX_H / h))
        with open(os.path.join(YOLO, 'labels', sub, bid + '.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    yaml = os.path.join(YOLO, 'solder.yaml')
    with open(yaml, 'w', encoding='utf-8') as f:
        f.write('path: %s\n' % YOLO.replace('\\', '/'))
        f.write('train: images/train\nval: images/val\ntest: images/test\n')
        f.write('names:\n  0: solder_joint\n')
    print('yolo dataset ready ->', yaml)


if __name__ == '__main__':
    main()
