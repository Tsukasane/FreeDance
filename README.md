# Group Dance

## Installation
```
pip install -r requirement.txt
```
* for fid calculation, use numpy==1.24.4
* for tensorboard usage, use protobuf==4.25.3
* for pytorch3d, follow the instruction [here]()

## Data Preparation
1. Dataset preprocessing
[TODO] add EDGE preprocessing scripts

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
    --num-local-layer 2
```

## Acknowledgement
We thank [EDGE]() and [MMM]() for their awesome codebases.