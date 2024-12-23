# Group Dance

## Installation
```
# install Anaconda / miniconda before running the scripts
conda env create -f environment.yml
```
* For fid calculation, use numpy==1.24.4
* For tensorboard usage, use protobuf==4.25.3
* If you encounter problems in installing pytorch3d, please consider follow the instruction [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-wheels-for-linux)

## Data Preparation
1. Dataset preprocessing

[TODO] add EDGE preprocessing scripts of AISTpp

2. Collect data statistics
```
# aistpp
python -m dataset.stat_collect
```

3. FID Feature Extractor (Motion AE Training)
```
# only using aistpp data to train, should we include data from other dataset later?
python -m eval.train
```

## Two-stage training
```
# multi person vqvae 
python train_vq.py --dataname aistpp --exp-name vq_dance_train

# music-motion transformer
python train_m2d_trans.py \
    --dataname aistpp \
    --vq-name 2024-12-21-03-47-22_vq_dance_train \
    --out-dir output/m2d \
    --exp-name trans_name \
    --num-local-layer 2 \
    --resume-trans output/m2d/2024-12-23-03-59-17_trans_m2d/net_last.pth
```

Use argument ``--resume-pth`` / ``--resume-trans`` to resume training vqvae / transformer.

## Acknowledgement
We thank [EDGE](https://github.com/Stanford-TML/EDGE) and [MMM](https://github.com/exitudio/MMM/) for their awesome codebases.