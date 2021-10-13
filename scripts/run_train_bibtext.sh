python train.py \
    --data_dir ../data/bibtext/bibtext_data.npy \
    --train_idx ../data/bibtext/bibtext_train_idx.npy \
    --valid_idx ../data/bibtext/bibtext_val_idx.npy \
    --test_idx ../data/bibtext/bibtext_test_idx.npy \
    --learning_rate 0.00025 \
    --max_epoch 200 \
    --meta_offset 0 \
    --label_dim 159 \
    --z_dim 159 \
    --feat_dim 1836 \
    --model_dir model/model_bibtext/ \
    --summary_dir summary/summary_bibtext/ \
    --visual_dir visualization/visualization_bibtext/ \
    --nll_coeff 10.0 \
    --l2_coeff 1.0 \
    --c_coeff 0.1 \
    --batch_size 128 \
    --test_sh_path ./run_test_bibtext.sh \
    --write_to_test_sh True \
    --dataname bibtext \
    --lr_decay_ratio 0.8 \
    --lr_decay_times 4. \
    --check_freq 20 \
    --keep_prob 0.5
