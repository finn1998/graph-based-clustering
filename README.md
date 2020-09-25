# Learning to Cluster Faces

This repo provides an official implementation for [1, 2] and a re-implementation of [3].

## Paper
1. [Learning to Cluster Faces on an Affinity Graph](https://arxiv.org/abs/1904.02749), CVPR 2019 (**Oral**) [[Project Page](http://yanglei.me/project/ltc)]
2. [Learning to Cluster Faces via Confidence and Connectivity Estimation](https://arxiv.org/abs/2004.00445), CVPR 2020 [[Project Page](http://yanglei.me/project/ltc_v2)]
3. [Linkage-based Face Clustering via Graph Convolution Network](https://arxiv.org/abs/1903.11306), CVPR 2019


## Requirements
* Python >= 3.6
* PyTorch >= 0.4.0
* [faiss](https://github.com/facebookresearch/faiss)
* [mmcv](https://github.com/open-mmlab/mmcv)


## Setup and get data


## Run
Run code with VEGCN:
* Modify input data in './vegcn/configs/cfg_train_gcnv_ms1m.py'
* Running script: sh scripts/vegcn/train_gcn_v_ms1m.sh


Follow the instructions in [dsgcn](dsgcn/), [vegcn](vegcn/) and [lgcn](lgcn/) to run algorithms.

