# QC-StyleGAN - Quality Controllable Image Generation and Manipulation
## Requirements

## Model Zoo

## Usage
### Image restoration
First, run pSp to get the initial latent codes for PTI. To do so, move to the `restoration/pSp` folder and run the following code:
```
python scripts/inference.py \
	--out_path=results/multi_deg_celebA \
	--checkpoint_path=../..pretrained_models/psp_ffhq.pt \
	--data_path=../../testsets/multi_deg/celebA/deg \
    --stylegan_weights=../../pretrained_models/qcstylegan_ffhq.pkl \
	--test_batch_size=4 \
	--test_workers=4 \
```
where `--checkpoint path` and `--stylegan_weights` is the provided pretrained pSp and QC-StyleGAN models, respectively (see the model zoo section).

After running the above script, move the the `restoration/PTI` folder and run the following code:
```
python inversion.py --network ../../pretrained_models/qcstylegan_ffhq.pkl \
		    --image_dir ../pSp/results/multi_deg_celebA/org \
		    --save_dir results/multi_deg_celebA \
		    --latent_dir ../pSp/results/multi_deg_celebA/latent \
		    --gen_degraded
```

### Image editing
First, follow the image restoration section to generate inverted latent codes and modified model weights (since we use PTI for image inversion). Then run the following script:
```
python edit.py \
    -i ../restoration/pSp/results/multi_deg_celebA/latent \
    -m ../restoration/PTI/results/multi_deg_celebA/models \
    -b boundaries/smiling_boundary.npy \
    -o results/smiling \
    -s W \
```
where `-b` is the path of the editing boundary (check interfaceGAN paper for more information), `-i` is the root of the latent codes genenrated in pSp section, `-m` is the root of the modified QC-StyleGAN weights in PTI section.
