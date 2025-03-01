# Group Dance

## TODOs
- [ ] Report the results of our method, ablations, comparison methods, 1D codebook baseline. 
- [ ] A README file for 1D baseline, including aioz preprocessing.
- [ ] A README file for comparison methods, including network adjustment, data adaptation, training parameters and time.
- [ ] Paper & supplementary material writing & revising.
- [ ] Code sanity check.
- [ ] Plot & blender visualization.
- [ ] Gradio demo & project page.
- [ ] Last check for paper submission.

    
## Installation
```
# install Anaconda / miniconda before running the scripts
conda env create -f environment.yml
```
* For fid calculation, use ``numpy==1.24.3``.
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

    *NOTE:* The partition follows the original manner.


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
python train_vq.py \
    --dataname aamixed \
    --exp-name vq_multi2d_aamixed1 \
    --vis-dir vq_multi2d_aamixed1 \
    --out-dir <your_folder> \
    --max-person 3 \
    --nb-code 4096 \
    --lr 5e-4 \
    --lr-scheduler 200000 \
    --resume-pth <your_checkpoint_path>

# music-motion transformer
python train_m2d_trans.py \
    --dataname aamixed \
    --vq-dir <stage1_output_folder> \
    --out-dir <stage2_output_folder> \
    --exp-name trans_multi2d_aamixed \
    --nb-code 4096 \
    --lr 5e-4 \
    --lr-scheduler 20 30 \
    --num-local-layer 2 \
    --resume-trans <your_checkpoint_path>
```

Use argument ``--resume-pth`` / ``--resume-trans`` to resume training vqvae / transformer.


## Ablation
* For 1D codebook, please check [this branch](https://github.com/Tsukasane/Group-Dance/tree/multi_baseline).
* For Stage 2 module design, please check ``./models/m2d_trans.py`` and modify bool variable ``use_moduleA``, ``use_moduleB``.


## Evaluation
We use FID (based on a pretrained motion autoencoder), diversity, and beat alignment score to evaluate model performance.

```
python evaluation.py \
    --resume-pth 'path/to/stage1/vq/net_last.pth' \
    --resume-trans 'path/to/stage2/transformer/net_last.pth' \
    --nb-code 4096 
```


## Visualization
* The skeleton video is automatically produced along the training of stage2.
* If you would like to see the retargeted character animation, please follow [SMPL-to-FBX installation](./SMPL-to-FBX/README.md). 

```
# check intermediate results of stage 1
python recons.py \
    --dataname aamixed \
    --exp-name vq_recons \
    --out-dir <your_path>
```


## Inference

For customized music inference and gradio demo.
```
python generate.py \
    --resume-pth 'path/to/stage1/vq/net_last.pth' \
    --resume-trans 'path/to/stage2/transformer/net_last.pth' \
    --nb-code 4096 \
    --music_dir 'folder/to/customized/music' \
    --cache_features \
    --feature_cache_dir 'folder/to/save/or/load/cached/music/feature' \
    --use_cached_features
        
```

/data/xingqunqi/AI_dance/Group_Dance_output/output/m2d/2025-02-22-11-37-52_trans_nb4096_newModuleBModuleAgamma02/net_best_fid.pth



*NOTE:* We set ``mask_logits=True`` in ``./models/m2d_trans.py`` at inference time to further ensure the predicted tokens are from the same codebook partition.

* ``--cache_features`` will save intermediate music features.
* `` --use_cached_features`` -- if specified, will not use the raw music but the preextracted features. Please also specify ``--feature_cache_dir``.
* Then the generate results will be saved under ``./inference_out``.


## Acknowledgement
We thank the awesome codebases, [EDGE](https://github.com/Stanford-TML/EDGE), [MMM](https://github.com/exitudio/MMM/), [Lodge](https://github.com/li-ronghui/LODGE) and [SMPL-to_FBX](https://github.com/softcat477/SMPL-to-FBX); and the helpful platform, [Blender](https://www.blender.org/).