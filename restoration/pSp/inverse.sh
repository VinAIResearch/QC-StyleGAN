python scripts/inference.py \
	--out_path=results/multi_deg_celebA \
	--checkpoint_path=../..pretrained_models/psp_ffhq.pt \
	--data_path=../../testsets/multi_deg/celebA/deg \
    	--stylegan_weights=../../pretrained_models/qcstylegan_ffhq.pkl \
	--test_batch_size=4 \
	--test_workers=4 \
