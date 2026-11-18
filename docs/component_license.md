# 复用组件清单与许可证说明

本清单对应交付包内 `solder_inspect_agent/` 的实际运行环境。版本号取自本机 `importlib.metadata` 与各包 `*.dist-info` 内的 `METADATA` / `licenses` 文件，许可证字段取自 `License-Expression`（无该字段时取自包内 LICENSE 首段），未使用推测值。

## 一、第三方软件组件

| 序号 | 组件 | 实测版本 | 在本项目中的作用 | 许可证 | 依据与来源 |
|---|---|---|---|---|---|
| 1 | NumPy | 2.5.5 | 数组运算、HOG/LBP 特征实现、指标计算 | BSD-3-Clause（表达式另含 0BSD、MIT、Zlib、CC0-1.0） | 包内 `numpy-2.5.5.dist-info/licenses/LICENSE.txt`；https://numpy.org |
| 2 | pandas | 3.1.1 | 标签表读写、划分表与指标汇总 | BSD-3-Clause | `METADATA` 的 `License-Expression`；https://pandas.pydata.org |
| 3 | scikit-learn | 1.9.1 | PCA 降维、RBF-SVM、GridSearchCV、混淆矩阵与分类指标 | BSD-3-Clause | 包内 `licenses/COPYING` 首行 “BSD 3-Clause License”；https://scikit-learn.org |
| 4 | opencv-python-headless | 5.1.0.95 | 图像读写、HSV 分割、形态学运算、轮廓提取、尺寸变换 | Apache-2.0（其内附 OpenCV 第三方声明） | `METADATA` 的 License 字段；包内 `LICENSE-3RD-PARTY.txt`；https://github.com/opencv/opencv-python |
| 5 | Matplotlib | 3.11.2 | 训练曲线、混淆矩阵、阈值曲线等图表输出 | PSF 风格许可证（BSD 兼容） | `METADATA` 的 License 字段；https://matplotlib.org |
| 6 | Pillow | 13.0.0 | ROI 裁剪、中文标注图绘制、图像格式转换 | MIT-CMU | `License-Expression: MIT-CMU`；包内 `LICENSE`；https://python-pillow.org |
| 7 | PyTorch | 2.15.0+cpu | 张量计算、模型训练与推理 | BSD-3-Clause（表达式为多许可证组合，含 Apache-2.0、MIT、BSD-2/3-Clause、BSL-1.0、ISC、Zlib） | `License-Expression`；包内 `LICENSE.txt`；https://pytorch.org |
| 8 | torchvision | 0.29.2+cpu | ResNet18 网络结构与 ImageNet 预训练权重、图像变换 | BSD-3-Clause | `METADATA` 的 License 字段；https://github.com/pytorch/vision |
| 9 | Ultralytics | 8.4.215 | YOLOv8n 检测器对照实验（最终未采用） | AGPL-3.0-only | `ultralytics_platform` 的 `License-Expression: AGPL-3.0-only`；https://ultralytics.com |
| 10 | Streamlit | 1.67.0 | 复核界面（上传、指标卡、改判记录） | Apache-2.0 | `License-Expression: Apache-2.0`；https://streamlit.io |
| 11 | python-docx | 1.2.0 | 仅用于生成课程报告 DOCX（不参与推理） | MIT | `METADATA` 的 License 字段；https://github.com/python-openxml/python-docx |

## 二、预训练权重

| 组件 | 来源 | 许可证 | 使用方式与限制 |
|---|---|---|---|
| ResNet18 ImageNet-1k 权重（`resnet18-f37072fd.pth`） | torchvision `ResNet18_Weights.IMAGENET1K_V1`，下载自 `download.pytorch.org` | BSD-3-Clause（随 torchvision 仓库发布） | 仅作为特征初始化，在本项目数据上微调；权重文件不随交付包分发，由脚本首次运行时自动下载 |

YOLOv8n 官方预训练权重在本机网络环境下无法获取（`github.com` 连接被重置），因此未纳入任何交付物；`ultralytics` 仅在对照实验阶段作为运行时依赖，其 AGPL-3.0 条款不覆盖本项目自写的定位、分类、路由与界面代码，交付仓库也不打包任何 `.pt` 权重文件。

## 三、数据集与素材

| 材料 | 来源 | 说明 |
|---|---|---|
| `data/raw`、`data/roi`、`labels_phone.csv`、`labels_scan.csv` | 本项目 `src/synth_data.py` 程序化合成 | 不含任何第三方图像、不含真实人员信息，可完全复现（固定随机种子），无版权与隐私风险 |
| 报告与图表 | 本项目实验输出 | 全部由 `results/` 下的真实运行产物生成，无外部截图 |

## 四、代码来源声明

| 模块 | 来源 | 说明 |
|---|---|---|
| `src/features.py` 的 HOG 实现 | 按 Dalal 与 Triggs（2005）提出的梯度方向直方图定义自行实现 | 未复制任何第三方实现代码；仅在注释中标注算法出处 |
| `src/features.py` 的 LBP 实现 | 按 Ojala 等（2002）的等价模式定义自行实现 | 同上 |
| `src/classify.py` 的 `build_resnet18` | 调用 `torchvision.models.resnet18` 官方实现 | 未修改网络结构，仅替换分类头 |
| `src/locate.py` 的 `ClassicLocator` | 本项目原创 | HSV 双通道分割与形态学后处理参数由本项目网格搜索确定 |
| `app/streamlit_app.py` | 本项目原创，使用 Streamlit 公开 API | 无第三方模板 |

## 五、字体与文档

报告 DOCX/PDF 使用的宋体、黑体、Times New Roman 均为本机操作系统已安装字体，交付包不嵌入、不再分发字体文件。

## 六、合规结论

1. 交付仓库的全部自写代码以 MIT 许可证发布（见 `LICENSE`），不含 GPL/AGPL 代码片段。
2. 所有第三方依赖均为宽松许可证（BSD、MIT、Apache-2.0、PSF），允许学术与商业使用；唯一的 AGPL 组件 Ultralytics 未进入最终推理链路，也未随包分发权重。
3. 未使用任何未经授权的数据集、图像或他人代码。
