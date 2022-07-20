# PseCo (ECCV 2022)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2203.16317)
![visitors](https://visitor-badge.glitch.me/badge?page_id=ligang-cs/PseCo)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pseco-pseudo-labeling-and-consistency/semi-supervised-object-detection-on-coco-100)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-100?p=pseco-pseudo-labeling-and-consistency)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pseco-pseudo-labeling-and-consistency/semi-supervised-object-detection-on-coco-10)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-10?p=pseco-pseudo-labeling-and-consistency)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pseco-pseudo-labeling-and-consistency/semi-supervised-object-detection-on-coco-5)](https://paperswithcode.com/sota/semi-supervised-object-detection-on-coco-5?p=pseco-pseudo-labeling-and-consistency)
### **PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection**

[Gang Li](https://scholar.google.com/citations?user=hrUHmEQAAAAJ&hl=en&oi=ao), [Xiang Li](http://implus.github.io/), [Yujie Wang](https://scholar.google.com/citations?user=7CobseIAAAAJ&hl=en&oi=ao), [Yichao Wu](https://scholar.google.com/citations?user=20Its9kAAAAJ&hl=en&oi=ao), [Ding Liang](https://scholar.google.com/citations?user=Dqjnn0gAAAAJ&hl=en&oi=ao), [Shanshan Zhang](https://sites.google.com/site/shanshanzhangshomepage/).

<img src=./docs/framework.png width="70%">

This repo is the official implementation of ECCV2022 paper ["PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection"](https://arxiv.org/abs/2203.16317). PseCo delves into two key techniques of semi-supervised learning (e.g., pseudo labeling and consistency training) for SSOD, and integrate object detection properties into them.

## 🧪Main Results

### Partial Labeled Data
Following the common practice, all experimental results are averaged  on 5 different data folds. 

#### 1% labeled data
| Method | mAP| Model Weights |
| ---- | -------| ----- |
| Supervised Baseline|  12.20 |-|
| PseCo| 22.43 |[BaiduYun](https://pan.baidu.com/s/1VIc3BnzxD_s0E_qsPHyiTA?pwd=nr12) |

#### 2% labeled data
| Method | mAP| Model Weights |
| ---- | -------| ----- |
| Supervised Baseline| 16.53 |-|
| PseCo| 27.77 | [BaiduYun](https://pan.baidu.com/s/1prSLnIlFUceMCwMBqxJg2w?pwd=u5jg) |

#### 5% labeled data
| Method | mAP| Model Weights |
| ---- | -------| ----- |
| Supervised Baseline|  21.17 |-|
| PseCo| 32.50 | [BaiduYun](https://pan.baidu.com/s/1VjBWFc7idZkjfvzIfo55IA?pwd=t8hu) |

#### 10% labeled data
| Method | mAP| Model Weights |
| ---- | -------| ----- |
| Supervised Baseline| 26.90 |-|
| PseCo| 36.06 |[BaiduYun](https://pan.baidu.com/s/1564lArFjzdTff_CetxYTnA?pwd=dwzm) |


### Full Labeled Data
| Method | mAP| Model Weights |
| ---- | -------| ----- |
| Supervised Baseline| 41.0 |-|
| PseCo| 46.1 |[BaiduYun](https://pan.baidu.com/s/1G8qUzfD6tVTSF35iiCJuQQ?pwd=i63y)|

## ➡️Usage
Since this repo is built on the [Soft Teacher](https://github.com/microsoft/SoftTeacher), some setup instructions are cloned from it.

### Requirements
- `Ubuntu 16.04`
- `Anaconda3` with `python=3.6`
- `Pytorch=1.9.0`
- `mmdetection=2.16.0+fe46ffe`
- `mmcv=1.3.9`

### Installation
```
pip install -r requirements.txt
cd thirdparty/mmdetection && pip install -e .
cd ../.. && pip install -e .
```

### Data Preparation
- Download the COCO dataset
- Execute the following command to generate data set splits:
```shell script
# YOUR_DATA should be a directory contains coco dataset.
# For eg.:
# YOUR_DATA/
#  coco/
#     train2017/
#     val2017/
#     unlabeled2017/
#     annotations/
ln -s ${YOUR_DATA} data
bash tools/dataset/prepare_coco_data.sh conduct
```
For concrete instructions of what should be downloaded, please refer to `tools/dataset/prepare_coco_data.sh` line [`11-24`](https://github.com/microsoft/SoftTeacher/blob/863d90a3aa98615be3d156e7d305a22c2a5075f5/tools/dataset/prepare_coco_data.sh#L11)
### Training
- To train model on the **partial labeled data** setting:

For 5% and 10% labelling ratios:
```shell script
bash tools/dist_train_partially_labeled.sh
```
While for 1% and 2% labelling ratios, half of training iterations are enough: 
```shell script
bash tools/dist_train_partially_labeled_90k_iter.sh
```
- To train model on the **full labeled data** setting:

```shell script
bash tools/dist_train_fully_labeled.sh 
```
All experiments are trained on 8 GPUs by default.


### Evaluation
```
bash tools/test.sh
```
Please specify your config and checkpoint path in the tools/test.sh.

## 🧱To-do List
- [x] Release PseCo codes.
- [ ] Apply PseCo to the one-stage detector.
- [ ] Release codes of our latest semi-supervised object detection method: [DTG](https://arxiv.org/abs/2207.05536).

## 👍Acknowledgement
We would like to thank the authors of [Soft Teacher](https://github.com/microsoft/SoftTeacher) and [mmdetection](https://github.com/open-mmlab/mmdetection). 

## ✏️Citation
Consider cite PseCo in your publication if it helps your research.
```bib
@article{li2022pseco,
  title={PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection},
  author={Li, Gang and Li, Xiang and Wang, Yujie and Zhang, Shanshan and Wu, Yichao and Liang, Ding},
  journal={arXiv preprint arXiv:2203.16317},
  year={2022}
}
```
