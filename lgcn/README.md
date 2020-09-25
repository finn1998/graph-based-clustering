# Linkage-based Face Clustering via GCN

## Main modification:

- refactory the training and testing with `Runner` from [mmcv](https://github.com/open-mmlab/mmcv), which is modularized for easy extension.
- replace `torch.Tensor` with `np.ndarray` in dataloader, which makes the dataloader usable in frameworks in addition to PyTorch.
- evaluate lgcn under the same setting as dsgcn.

## Test

Download the pretrained models in the [model zoo](https://github.com/yl-1993/learn-to-cluster/blob/master/MODEL_ZOO.md).

Test

```bash
# Testing takes about 3 hours on 1 TitanX.
sh scripts/lgcn/test_lgcn_ms1m.sh
```

## Train

We use the training parameters with best performance in our experiments as the default config.

```bash
# Training takes about 27 hours on 1 TitanX.
sh scripts/lgcn/train_lgcn_ms1m.sh
```

If there is better training config, you are welcome to report to us. 

## Reference

- Paper: https://arxiv.org/abs/1903.11306
- Code: https://github.com/Zhongdao/gcn_clustering
