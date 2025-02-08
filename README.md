# Group Dance

## TODOs (please check the experiment sheet in group chat)
- [ ] **(24/2/1 - 25/2/7 ongoing)** 2D codebook, 1-3 person, on aamixed dataset (Yiwen).
- [ ] **(24/2/1 - 25/2/7 ongoing)** 2D codebook, 1-3 person, ModuleA(reaction attention), on aamixed dataset (Yiwen).
- [ ] **(24/2/1 - 25/2/7 ongoing)** 2D codebook, 1-3 person, ModuleB(temporal coherent cross-attention), on aamixed dataset (Yiwen).
- [ ] **(24/2/1 - 25/2/7 ongoing)** 2D codebook, 1-3 person, ModuleA+B(temporal coherent cross-attention), on aamixed dataset (Yiwen).
- [ ] **(24/2/3 - 25/2/7)** 1D codebook, 1-3 person, baseline on aistpp & aamixed dataset (Liting). 
    * Please also add detailed steps to the README in your branch.
    * Save the checkpoints corresponding to your results.
    * List all the parameters numbers you tuned.
- [ ] **(25/2/3 - 25/2/17)** Openresource codebase search & Run Comparison Methods on aamixed dataset (Yang). 
    * 1+ group dance, 3-4 single person dance but switch to group dance by simply adding more dimensions, 2 text-to-motion but switch text feature and encoder to music.
    * Yiwen will provide the paper list.
- [ ] **(25/2/7 - 25/2/14)** MoE design and implementation (Yiwen).
- [ ] **(25/2/14 - 25/3/6)** Paper & supplementary material writing & revising (Yiwen, Xingqun).
- [ ] **(25/2/14 - 25/3/6)** Code sanity check (Xingqun).
- [ ] **(25/2/17 - 25/3/6)** Plot & blender visualization (Yang).
    * Yiwen will provide drafts.
    * The rough visualization tutorial is in dev branch.
- [ ] **(25/3/1 - 25/3/6)** Gradio demo & project page. (Yiwen)
- [ ] **(25/3/6 - 25/3/7)** Last check for paper submission. (All)

    
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
* [aist++](https://github.com/google/aistplusplus_api/tree/main)
    ```.bash
    cd preprocess/aistpp
    bash download_dataset.sh
    python create_dataset.py --extract-baseline --dataset_folder <your_folder>
    ```

* [AIOZ-GDance](https://github.com/aioz-ai/AIOZ-GDANCE?tab=readme-ov-file#aioz-gdance-dataset)
    ```
    ```

* Mixed

    We combine the above two datasets to train our model generating free-number of dancers. Since there predefined ground planes are different, a alignment transition is calculated using

    ```
    python -m dataset.stat_collect_multi --stage 1
    ```

    This is the ``delta_height`` we specified in ``./dataset/dataset_MD_multi.py``
    
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

    Specify the data statistics saving path in ``./dataset/stat_collect_multi.py``
    ```
    # for aistpp
    python -m dataset.stat_collect_multi --stage 2 --dataset_name aistpp

    # for aioz
    python -m dataset.stat_collect_multi --stage 2 --dataset_name aioz

    # for aamixed
    python -m dataset.stat_collect_multi --stage 2 --dataset_name aamixed
    ```

3. FID Feature Extractor (Motion AE Training)
    ```
    # using aamixed data
    python -m eval_legacy.train \
        --dataset_name aamixed \
        --checkpoint_dir ./eval_legacy/checkpoints_aamixed

    # aistpp only
    python -m eval_legacy.train \
        --dataset_name aistpp \
        --checkpoint_dir ./eval_legacy/checkpoints_aistpp \
        --epochs 200
    ```


## Two-stage training
```
# multi person vqvae 
CUDA_VISIBLE_DEVICES=4 python train_vq.py \
    --dataname aamixed \
    --exp-name vq_multi2d_aamixed1 \
    --vis-dir vq_multi2d_aamixed1 \
    --out-dir /data/xingqunqi/AI_dance/Group_Dance_output/output \
    --resume-pth /data/xingqunqi/AI_dance/Group_Dance_output/output/vq/2025-01-28-23-57-52_vq_multi2d_aamixed1/net_last.pth


# music-motion transformer
CUDA_VISIBLE_DEVICES=5 python train_m2d_trans.py \
    --dataname aamixed \
    --vq-dir /data/xingqunqi/AI_dance/Group_Dance_output/output/vq/te2 \
    --out-dir /data/xingqunqi/AI_dance/Group_Dance_output/output/m2d \
    --exp-name trans_multi2d_aamixed \
    --num-local-layer 2 \
    --resume-trans output/m2d/2024-12-24-06-43-28_trans_name/net_last.pth

```

Use argument ``--resume-pth`` / ``--resume-trans`` to resume training vqvae / transformer.


## Ablation
* For 1D codebook, please check [this branch](https://github.com/Tsukasane/Group-Dance/tree/multi_baseline).
* For Stage 2 module design, please check ``./models/m2d_trans.py`` and modify bool variable ``use_moduleA``, ``use_moduleB``.


## Evaluation
First, extract the statistical kinetic and manual features of a mixed dataset. Currently, this process is automatically performed when the data is the first time passing the data loader. Please note that it will cause the first pass to be extremely slow. You can modify ``./dataset/dataset_MD_multi.py`` to disable this step.

```
# check intermediate results of stage 1
CUDA_VISIBLE_DEVICES=1 python recons.py \
    --dataname aamixed \
    --exp-name vq_recons \
    --out-dir /data/xingqunqi/AI_dance/Group_Dance_output/output
```

Then, calculate the metrics (FID, Dist, Beat...) of new generated dance.
```
python eval/calculate_scores.py
python eval/calculate_beat_scores.py
```


## Visualization
* The skeleton video is produced along the training.
* If you would like to see the retargeted character animation, please follow [SMPL-to-FBX installation](./SMPL-to-FBX/README.md). 


## Inference

For customized music inference and gradio demo.
```
CUDA_VISIBLE_DEVICES=5 python generate.py \
        --resume-pth '/data/xingqunqi/AI_dance/Group_Dance_output/output/vq/2025-01-29-10-18-55_vq_multi2d_aamixed1/net_last.pth' \
        --resume-trans '/data/xingqunqi/AI_dance/Group_Dance_output/output/m2d/2025-01-30-05-15-01_trans_multi2d_aamixed_te3_60000st/net_last.pth' \
        --music_dir '/home/xingqunqi/AI_dance/AI_dance/demos/group-dance-demo/resources' \
        --cache_features \
        --feature_cache_dir '/home/xingqunqi/AI_dance/AI_dance/inference_music_feats' \
        --use_cached_features
        
```

* ``--resume-pth`` -- the vqvae checkpoint.
* ``--resume-trans`` -- the transformer checkpoint.
* ``--music_dir`` -- dir for music segments.
* ``--cache_features`` will save intermediate music features under ``./inference``.
* `` --use_cached_features`` -- if specified, will not use the raw music but the preextracted features. Please also specify ``--feature_cache_dir``.
* Then the generate results will be saved under ``./inference_out``.


## Acknowledgement
We thank the awesome codebases, [EDGE](https://github.com/Stanford-TML/EDGE), [MMM](https://github.com/exitudio/MMM/), [POPDG](https://github.com/Luke-Luo1/POPDG/) and [SMPL-to_FBX](https://github.com/softcat477/SMPL-to-FBX); and the helpful platform, [Blender](https://www.blender.org/).