# Group Dance

two-stage training
```
CUDA_VISIBLE_DEVICES=7 python train_vq.py --dataname aistpp --exp-name vq_debug --total-iter 30000 --warm-up-iter 200

CUDA_VISIBLE_DEVICES=7 python train_vq.py --dataname kit --exp-name vq_name --total-iter 30000 --warm-up-iter 200

CUDA_VISIBLE_DEVICE=7 python train_t2m_trans.py --dataname kit --vq-name 2024-12-02-03-46-11_vq_name --out-dir output/t2m --exp-name trans_name --num-local-layer 2

CUDA_VISIBLE_DEVICE=7 python train_m2d_trans.py --dataname aistpp --vq-name 2024-12-01-23-08-29_vq_debug --out-dir output/t2m --exp-name trans_name --num-local-layer 2
```

Collect data statistics
```
python -m dataset.stat_collect

```

Fid Feature Extractor
```
# followed MMM motion feature extractor
CUDA_VISIBLE_DEVICES=7 python -m eval.train
```