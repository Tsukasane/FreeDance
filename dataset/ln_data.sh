#!/bin/bash
PATH_TO_AISTPP=/Your_Path/to/AIST++_dataset
PATH_TO_AAMIXED=/Your_Path/to/aamixed_dataset
PATH_TO_AIOZ=/Your_Path/to/AIOZ_Gdance_dataset


for dset in "train" "val" "test"; do
    for feats in "wavs_sliced" "motions_sliced" "baseline_feats"; do
        mkdir -p "${PATH_TO_AAMIXED}/${dset}/${feats}"
    done
done

#aistpp
for dset in "train" "test"; do
    echo processing "${PATH_TO_AISTPP}/${dset}"
    for file in ${PATH_TO_AISTPP}/${dset}/wavs_sliced/*; do
        ln -s "$file" ${PATH_TO_AAMIXED}/${dset}/wavs_sliced/
    done

    for file in ${PATH_TO_AISTPP}/${dset}/motions_sliced/*; do
        ln -s "$file" ${PATH_TO_AAMIXED}/${dset}/motions_sliced/
    done

    for file in ${PATH_TO_AISTPP}/${dset}/baseline_feats/*; do
        ln -s "$file" ${PATH_TO_AAMIXED}/${dset}/baseline_feats/
    done
done

#aioz
for dset in "train" "val" "test"; do
    echo processing "${PATH_TO_AIOZ}/${dset}"
    for file in ${PATH_TO_AIOZ}/${dset}/wavs_sliced/*; do
        ln -s "$file" ${PATH_TO_AAMIXED}/${dset}/wavs_sliced/
    done

    for file in ${PATH_TO_AIOZ}/${dset}/motions_sliced/*; do
        ln -s "$file" ${PATH_TO_AAMIXED}/${dset}/motions_sliced/
    done

    for file in ${PATH_TO_AIOZ}/${dset}/baseline_feats/*; do
        ln -s "$file" ${PATH_TO_AAMIXED}/${dset}/baseline_feats/
    done
done