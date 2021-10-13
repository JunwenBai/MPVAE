python test.py \
    --data_dir ../data/mirflickr/mirflickr_data.npy \
    --test_idx ../data/mirflickr/mirflickr_test_idx.npy \
    --label_dim 38 \
    --z_dim 38 \
    --feat_dim 1000 \
    --testing_size 64 \
    --dataname mirflickr \
    --checkpoint_path model/model_mirflickr/lr-0.00075_lr-decay_0.50_lr-times_4.0_nll-0.50_l2-1.00_c-10.00/model-456
