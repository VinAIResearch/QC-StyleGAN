import abc
import os
import wandb
import os.path
from criteria.localitly_regulizer import Space_Regulizer
import torch
from lpips import LPIPS
from configs import global_config, paths_config, hyperparameters
from criteria import l2_loss, my_loss
from utils.models_utils import toogle_grad, load_old_G


class BaseCoach:
    def __init__(self, data_loader, use_wandb):

        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0

        # self.style_loss = my_loss.StyleLoss().to(global_config.device)
        # self.adain_loss = my_loss.AdaINLoss().to(global_config.device)
        self.stats_loss = my_loss.StatsLoss().to(global_config.device)

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()

        # Initialize checkpoint dir
        self.checkpoint_dir = paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self):

        # Initialize networks
        self.G = load_old_G()
        toogle_grad(self.G, True)

        self.original_G = load_old_G()

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def get_inversion(self, w_path_dir, image_name, image):
        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not hyperparameters.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        return optimizer

    def calc_loss(self, generated_sharp_images, org_sharp_images, generated_deg_images, org_deg_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_deg_images, real_images)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_deg_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        # loss_percep = 0.5 * self.style_loss(generated_sharp_images, org_sharp_images)
        # loss += loss_percep

        # loss_stats = 0.5 * self.adain_loss(generated_sharp_images, org_sharp_images, generated_deg_images, org_deg_images)
        # loss_stats = 0.5 * self.adain_loss(generated_sharp_images, org_sharp_images)
        loss_stats = 1000 * self.stats_loss(generated_sharp_images, org_sharp_images, generated_deg_images, org_deg_images)
        loss += loss_stats
        # loss_percep = 0

        return loss, l2_loss_val, loss_lpips, loss_stats

    def forward(self, w, q):
        generated_images, _ = self.G.synthesis(w, q, noise_mode='const', force_fp32=True)

        return generated_images
