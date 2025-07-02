# FreeDance
FreeDance: Towards Harmonic Free-Number Group Dance Generation via a Unified Framework. (ICCV 2025)

## TODOs
- [ ] Clean & Sync Codebase
- [ ] Project Page
- [ ] Pipeline Figure and More Samples
- [ ] Description of Solo Dance Comparison Methods Adaptation

## Installation
```
# install Anaconda / miniconda before running the scripts
conda create -n freedance python=3.8 -y
conda activate freedance

# adjust to your own cuda version
conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .

cd ..
pip install -r requirements.txt
```

## Data Preparation
1. Dataset preprocessing
* [aist++](https://github.com/google/aistplusplus_api/tree/main)
    ```.bash
    cd preprocess/aistpp
    bash download_dataset.sh
    python create_dataset.py --extract-baseline --dataset_folder <your_folder>
    ```

* [AIOZ-GDance](https://github.com/aioz-ai/AIOZ-GDANCE?tab=readme-ov-file#aioz-gdance-dataset)
    ```.bash
    # Download dataset
    cd preprocess/aioz_gdance
    python create_dataset.py --extract-baseline --dataset_folder <your_folder>
    ```

* Mixed (aamixed)
    
    We combine the AIST++ and AIOZ-GDance datasets to train our model to generate a flexible number of dancers. You can symlink the processed data to ``./dataset/aamixed_dataset/`` by
    ```
    # fill the data path in ln_data.sh, then
    cd dataset
    bash ln_data.sh
    ```
    The structure is like

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
    
    An alignment transition (the ``delta_height`` we specified in ``./dataset/dataset_MD_multi.py``) of average pelvis position is calculated using
    ```
    python -m dataset.stat_collect_multi --stage 1
    ```


2. Collect data statistics

    Specify the data statistics saving path in ``./dataset/stat_collect_multi.py``
    ```
    # for aamixed
    python -m dataset.stat_collect_multi --stage 2 --dataset_name aamixed
    ```

3. FID Feature Extractor (Motion AE Training)
    ```
    # using aamixed data
    python -m eval_legacy.train \
        --dataset_name aamixed \
        --checkpoint_dir ./eval_legacy/checkpoints_aamixed
    ```


## Two-stage training
```
# multi dancers vq 
python train_vq.py \
    --dataname aamixed \
    --exp-name vq_<exp_name> \
    --vis-dir vis/vis_vq \
    --out-dir exp/vq_<exp_name> \
    --max-person 3 \
    --nb-code 4096 \
    --lr 5e-4 \
    --lr-scheduler 20000 50000 \
    --resume-pth <vq_checkpoint_path>

# music-motion masked token modeling
python train_m2d_trans.py \
    --dataname aamixed \
    --vq-dir exp/vq_<dir_name> \
    --out-dir exp/mtm_<dir_name> \
    --exp-name mtm_<exp_name> \
    --nb-code 4096 \
    --lr 5e-5 \
    --lr-scheduler 20 30 \
    --num-local-layer 2 \
    --resume-trans <trans_checkpoint_path>
```

Use argument ``--resume-pth`` / ``--resume-trans`` to resume training vqvae / MTM transformer.


## Evaluation
We use FID (based on a pretrained motion autoencoder), diversity, and beat alignment score as objective evaluation metrics.

```
python evaluation.py \
    --resume-pth 'path/to/stage1/vq/net_last.pth' \
    --resume-trans 'path/to/stage2/transformer/net_last.pth' \
    --nb-code 4096 
```


## Visualization
* The skeleton video is automatically produced along the training of stage2.
* If you would like to see the retargeted character animation, please follow [SMPL-to-FBX installation](./SMPL-to-FBX/README.md). 



## Inference

Inference and visualize results of customized music.
```
# use music waveform
python generate.py \
    --resume-pth 'path/to/stage1/vq/net_last.pth' \
    --resume-trans 'path/to/stage2/transformer/net_last.pth' \
    --nb-code 4096 \
    --music_dir 'folder/to/customized/music'

# use preextracted features
python generate.py \
    --resume-pth 'path/to/stage1/vq/net_last.pth' \
    --resume-trans 'path/to/stage2/transformer/net_last.pth' \
    --nb-code 4096 \
    --feature_cache_dir 'folder/to/save/or/load/cached/music/feature' \
    --use_cached_features
        
```

* ``--cache_features`` will save intermediate music features.
* `` --use_cached_features`` -- if specified, will not use the raw music but the preextracted features. Please also specify ``--feature_cache_dir``.
* The generate results will be saved under ``./results``.
* Temporarily only support music phrases in ``.wav`` format.


## Acknowledgement
We thank the awesome codebases, [EDGE](https://github.com/Stanford-TML/EDGE), [MMM](https://github.com/exitudio/MMM/), [Lodge](https://github.com/li-ronghui/LODGE), and [SMPL-to_FBX](https://github.com/softcat477/SMPL-to-FBX); and the helpful platform, [Blender](https://www.blender.org/).