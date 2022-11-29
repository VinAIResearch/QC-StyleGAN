python3 scripts/train.py \
	--dataset_type=afhq_cat_encode \
	--exp_dir=./logs/afhq_cat_w_encoder \
	--coach alternative \
	--workers=4 \
	--batch_size=4 \
	--test_batch_size=8 \
	--test_workers=8 \
	--val_interval=5000 \
	--save_interval=20000 \
	--encoder_type=DoubleEncoder \
	--start_from_latent_avg \
	--lpips_lambda=0.8 \
    --learn_in_w \
	--l2_lambda=1 \
	--id_lambda=0 \
	--output_size 256 \