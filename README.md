# Group Dance

## TODOs
- [ ] **(24/12/24 - 25/1/7 ongoing)** Multi-person CB design.
- [ ] **(24/12/24 - 25/1/10 ongoing)** AIOZ-GDance dataset baseline.
- [ ] Check FID eval.
- [ ] Modify ``GPT_eval_multi.py``, the eval script.
- [ ] Modify ``generate.py`` to support custom music inference.
    - Extract music features, load pretrained vqvae model and transformer encoder(w/ multi-person design) to inference motion seq.
    
## Installation
```
# install Anaconda / miniconda before running the scripts
conda env create -f environment.yml
```
* For fid calculation, use ``numpy==1.24.4``.
* For tensorboard usage, use ``protobuf==4.25.3``.
* If you encounter problems in installing ``pytorch3d``, please consider follow the instruction [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-wheels-for-linux).

## Data Preparation
1. Dataset preprocessing

```.bash
cd preprocess/aistpp
bash download_dataset.sh
python create_dataset.py --extract-baseline --dataset_folder <your_folder>
```

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
    --resume-trans output/m2d/2024-12-24-06-43-28_trans_name
```

Use argument ``--resume-pth`` / ``--resume-trans`` to resume training vqvae / transformer.

## Visualization
* The skeleton video is produced along the training.
* If you would like to see the retargeted character animation, please follow [SMPL-to-FBX installation](./SMPL-to-FBX/README.md). 

## Acknowledgement
We thank the awesome codebases, [EDGE](https://github.com/Stanford-TML/EDGE), [MMM](https://github.com/exitudio/MMM/), and [SMPL-to_FBX](https://github.com/softcat477/SMPL-to-FBX); and the helpful platform, [Blender](https://www.blender.org/).