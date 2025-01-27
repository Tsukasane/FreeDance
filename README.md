# Group Dance

## TODOs
- [x] Multi-person CB design.
- [x] **(25/1/1 - 25/1/14 ongoing)** Reaction Attention design and implementation.
- [x] **(25/1/1 - 25/1/14 ongoing)** Music alignment design and implementation.
- [ ] **(24/12/24 - 25/1/7 ongoing)** AIOZ-GDance dataset baseline.
- [ ] **(25/1/1 - 25/1/14 ongoing)** MoE design and implementation.
- [ ] **(25/1/14 - 25/2/14)** Main Experiments/Baseline Comparison.
    - [ ] Check eval scripts.
    - [ ] Modify ``GPT_eval_multi.py``, the eval script.
    - [x] Modify ``generate.py`` to support custom music inference.
- [ ] **(25/2/14 - 25/3/6)** Paper writing & revising.
- [ ] **(25/2/14 - 25/3/6)** Code sanity check.
- [ ] **(25/2/14 - 25/3/6)** Plot & visualization.
- [ ] **(25/3/6 - )** Gradio demo & project page.
    
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
* [aist++]()
    ```.bash
    cd preprocess/aistpp
    bash download_dataset.sh
    python create_dataset.py --extract-baseline --dataset_folder <your_folder>
    ```

* [AIOZ-GDance]()
    ```
    ```

* Mixed

    We combine the above two datasets to train our model generating free-number of dancers.
You can symlink the processed aist++ and aioz-gdance data to ``./dataset/aamixed_dataset/``
The structures are like

    ```
    aamixed_dataset
        |--test
        |   |--baseline_feats
        |   |--motions_sliced
        |   |--wavs_sliced
        |--val
        |   |--baseline_feats
        |   |--motions_sliced
        |   |--wavs_sliced
        |--train
            |--baseline_feats
            |--motions_sliced
            |--wavs_sliced
    ```

*NOTE:* The partition follows the original manner. aistpp has no validation set, so that the aamixed validation set only contains data from aioz.


2. Collect data statistics
Specify the data statistics you want to collect in ``./dataset/stat_collect_multi.py``.
    ```
    # run stage 1 to calculate the transition for dataset alignment
    # run stage 2 to collect the data stats
    python -m dataset.stat_collect_multi
    ```

3. FID Feature Extractor (Motion AE Training)
    ```
    # only using aistpp data to train, should we include data from other dataset later?
    python -m eval.train
    ```

## Two-stage training
```
# multi person vqvae 
python train_vq.py --dataname aistpp --exp-name vq_dance2d_train

# music-motion transformer
python train_m2d_trans.py \
    --dataname aistpp \
    --vq-name 2024-12-21-03-47-22_vq_dance_train \
    --out-dir output/m2d \
    --exp-name trans_name \
    --num-local-layer 2 \
    --resume-trans output/m2d/2024-12-24-06-43-28_trans_name/net_last.pth

```

Use argument ``--resume-pth`` / ``--resume-trans`` to resume training vqvae / transformer.

## Visualization
* The skeleton video is produced along the training.
* If you would like to see the retargeted character animation, please follow [SMPL-to-FBX installation](./SMPL-to-FBX/README.md). 


## Inference
```
CUDA_VISIBLE_DEVICES=0 python generate.py \
        --resume-pth './output/vq/2025-01-01-10-47-58_vq_dance2d_train/net_last.pth' \
        --resume-trans './output/m2d/2025-01-01-04-28-09_trans_2d/net_last.pth' \
        --music_dir './demos/group-dance-demo/resources/' \
        --cache_features \
        --feature_cache_dir '/home/xingqunqi/AI_dance/AI_dance/inference' \
        --use_cached_features
```
* ``--cache_features`` will save intermediate music features under ``./inference``.
* Then the generate results will be saved under ``./inference_out``.

## Acknowledgement
We thank the awesome codebases, [EDGE](https://github.com/Stanford-TML/EDGE), [MMM](https://github.com/exitudio/MMM/), [POPDG](https://github.com/Luke-Luo1/POPDG/) and [SMPL-to_FBX](https://github.com/softcat477/SMPL-to-FBX); and the helpful platform, [Blender](https://www.blender.org/).