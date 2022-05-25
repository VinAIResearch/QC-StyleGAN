## Architechture
lpips_type = 'alex'
first_inv_type = 'w'
optim_type = 'adam'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = True
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
# regulizer_lpips_lambda = 0.2
regulizer_alpha = 30

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1

## Steps
# LPIPS_value_threshold = 0.06
LPIPS_value_threshold = 0.002
max_pti_steps = 500
# max_pti_steps = 350
first_inv_steps = 5000
# first_inv_steps = 450
max_images_to_invert = 10

## Optimization
pti_learning_rate = 3e-4
first_inv_lr = 5e-3
q_inv_lr = 3e-2
w_opt_loss = 0.19
train_batch_size = 1
use_last_w_pivots = False

