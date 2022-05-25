# LATENT_CODE_NUM=10
# python edit.py \
# 	    -m stylegan_ffhq \
# 	        -b boundaries/stylegan_ffhq_gender_z_boundary.npy \
# 		    -n "$LATENT_CODE_NUM" \
# 		        -o results/stylegan_ffhq_gender_z

python edit.py \
    -i ../restoration/pSp/results/multi_deg_celebA/latent \
    -m ../restoration/PTI/results/multi_deg_celebA/models \
    -b boundaries/smiling_boundary.npy \
    -o results/smiling \
    -s W \
