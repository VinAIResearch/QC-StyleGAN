import argparse
from configs import paths_config, hyperparameters, global_config
from scripts.run_pti import run_PTI

#         torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Controller-GAN Inversion')
parser.add_argument('--image_dir', default='inputs/org', help='Image directory')
parser.add_argument('--latent_dir', default='inputs/latent', help='Image directory')
parser.add_argument('--save_dir', default='output', help='Save directory')
parser.add_argument('--gen_degraded', action="store_true", default=False, help='Generate degraded image')
parser.add_argument('--network', default='../training-runs/00002-ffhq256-paper256-gamma3-batch64-resumecustom/network-snapshot-013708.pkl', help='network')
parser.add_argument('--name', default="controllergan", help='Name')
parser.add_argument('--image_dir_name', default='image', help='image directory name')

args = parser.parse_args()

if __name__ == '__main__':
    image_name = args.name
    use_multi_id_training = False
    global_config.device = 'cuda'
    paths_config.input_data_id = args.image_dir_name
    paths_config.input_data_path = args.image_dir
    paths_config.input_latent_path = args.latent_dir
    paths_config.stylegan2_ada_ffhq = args.network
    hyperparameters.use_locality_regularization = False
    global_config.gen_degraded = args.gen_degraded
    global_config.save_dir = args.save_dir

    model_id = run_PTI(use_wandb=False, use_multi_id_training=use_multi_id_training)
