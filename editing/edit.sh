# LATENT_CODE_NUM=10
# python edit.py \
# 	    -m stylegan_ffhq \
# 	        -b boundaries/stylegan_ffhq_gender_z_boundary.npy \
# 		    -n "$LATENT_CODE_NUM" \
# 		        -o results/stylegan_ffhq_gender_z

python3 edit.py \
    -i /home/datnvt/01.Datasets/resultsx/multi_deg_celebA/latent \
    -m /home/datnvt/01.Datasets/resultsx/multi_deg_celebA/models \
    -b boundaries/smiling_boundary.npy \
    -o /home/datnvt/01.Datasets/resultsx/smiling \
    -s W \
