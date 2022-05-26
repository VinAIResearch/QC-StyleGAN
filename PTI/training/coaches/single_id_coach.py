import os
import torch
from PIL import Image
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w


class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
        self.gen_degraded = global_config.gen_degraded
        self.save_dir = global_config.save_dir

    def train(self):

        use_ball_holder = True

        for fname, image in tqdm(self.data_loader):
            image_name = fname[0]

            q = torch.randn(1, self.G.q_dim).to(global_config.device)

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            w_pivot = None

            if hyperparameters.use_last_w_pivots:
                w_pivot = self.load_inversions(w_path_dir, image_name)

            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot, q_opt = self.calc_inversions(image, image_name, q)

            w_pivot = w_pivot.to(global_config.device)
            q = q_opt.to(global_config.device)

            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            for i in tqdm(range(hyperparameters.max_pti_steps)):
                generated_images = self.forward(w_pivot, q)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [image_name])

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1
            
            degraded_image = sharp_image = None
            if self.gen_degraded:
                with torch.no_grad():
                    degraded_image, _ = self.G.synthesis(w_pivot, q, noise_mode='const', force_fp32 = True)
            
            with torch.no_grad():
                sharp_image, _ = self.G.synthesis(w_pivot, torch.zeros(1, self.G.q_dim).to(global_config.device), noise_mode='const', force_fp32 = True)
            
            if self.gen_degraded:
                degraded_image = (degraded_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
                resized_image = Image.fromarray(degraded_image, mode='RGB')
                save_path = os.path.join(self.save_dir, "degraded_gen", image_name + ".png")
                dir_name = os.path.dirname(save_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                resized_image.save(save_path)

            sharp_image = (sharp_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
            resized_image = Image.fromarray(sharp_image, mode='RGB')
            save_path = os.path.join(self.save_dir, "sharp_gen", image_name + ".png")
            dir_name = os.path.dirname(save_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            resized_image.save(save_path)
