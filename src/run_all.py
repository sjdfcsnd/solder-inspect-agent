import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ('数据生成', ['-m', 'src.synth_data'], 'data/roi'),
    ('训练 SVM', ['-m', 'src.train', '--only', 'svm'], 'weights/svm.pkl'),
    ('训练 ResNet18', ['-m', 'src.train', '--only', 'resnet18', '--epochs', '12'], 'weights/resnet18.pt'),
    ('训练浅层 CNN', ['-m', 'src.train', '--only', 'shallow', '--epochs', '40'], 'weights/shallow.pt'),
    ('评估', ['-m', 'src.evaluate'], 'results/eval.json'),
    ('单图推理', ['-m', 'src.agent', '--image', 'tests/samples/board_055.png',
                 '--classifier', 'resnet18', '--save', 'results/demo'], 'results/demo'),
]


def main():
    ap = argparse.ArgumentParser(description='按附录 B 顺序复现全流程')
    ap.add_argument('--rebuild', action='store_true', help='强制重建数据与评估结果（不重训模型）')
    ap.add_argument('--retrain', action='store_true', help='强制从零重训三段模型')
    ap.add_argument('--skip-train', action='store_true', help='跳过三段训练，只跑数据与评估')
    args = ap.parse_args()

    t_all = time.time()
    for name, argv, marker in STEPS:
        is_train = argv[0] == '-m' and argv[1] == 'src.train'
        if is_train:
            run = args.retrain and not args.skip_train
        else:
            run = args.rebuild or not os.path.exists(os.path.join(ROOT, marker))
        if not run:
            print('[skip] %s (已有 %s)' % (name, marker))
            continue
        print('[run ] %s: python %s' % (name, ' '.join(argv)))
        t0 = time.time()
        rc = subprocess.run([sys.executable] + argv, cwd=ROOT).returncode
        if rc != 0:
            print('[fail] %s 退出码 %d' % (name, rc))
            return rc
        print('[done] %s %.1fs' % (name, time.time() - t0))
    print('全部步骤完成，总耗时 %.1fs' % (time.time() - t_all))
    return 0


if __name__ == '__main__':
    sys.exit(main())
