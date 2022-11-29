python3 scripts/inference.py \
	--out_path=/home/datnvt/01.Datasets/resultsx/multi_deg_celebA \
	--checkpoint_path=/home/datnvt/02.Codes/QC-StyleGAN-checkpoints/ffhq_psp.pt \
	--data_path=/home/datnvt/01.Datasets/testsets/multi_deg/celebA/deg \
    	--stylegan_weights=/home/datnvt/02.Codes/QC-StyleGAN-checkpoints/ffhq_256x256.pkl \
	--test_batch_size=4 \
	--test_workers=4 \
