python train.py \
    --data_dir ../data/yeast/yeast_data.npy \
    --train_idx ../data/yeast/yeast_train_idx.npy \
    --valid_idx ../data/yeast/yeast_val_idx.npy \
    --test_idx ../data/yeast/yeast_test_idx.npy \
    --learning_rate 0.0006 \
    --max_epoch 800 \
    --meta_offset 0 \
    --label_dim 14 \
    --z_dim 14 \
    --feat_dim 103 \
    --model_dir model/model_yeast/ \
    --summary_dir summary/summary_yeast/ \
    --visual_dir visualization/visualization_yeast/ \
    --nll_coeff 0.1 \
    --l2_coeff 1.0 \
    --c_coeff 20. \
    --batch_size 128 \
    --test_sh_path ./run_test_yeast.sh \
    --write_to_test_sh True \
    --dataname yeast \
    --lr_decay_ratio 0.8 \
    --lr_decay_times 4. \
    --check_freq 5 \
    --keep_prob 0.5
