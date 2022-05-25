import os
import numpy as np
import time
import torch
from PIL import Image
from tqdm import tqdm
from configs import hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w


class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
        self.gen_degraded = global_config.gen_degraded
        self.save_dir = global_config.save_dir

    def train(self):

        use_ball_holder = True
        runtimes = []

        for fname, image, latent in tqdm(self.data_loader):
            image_name = fname[0]

            q = torch.randn(1, self.G.q_dim).to(global_config.device)

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            start = time.time()
            w_pivot = None

            w_pivot = latent[:, :512].unsqueeze(1).repeat(1, self.G.num_ws, 1).to(global_config.device)
            q = latent[:, 512:].to(global_config.device)

            org_deg_images, _ = self.original_G.synthesis(w_pivot, q, noise_mode='const', force_fp32=True)
            org_sharp_images, _ = self.original_G.synthesis(w_pivot, torch.zeros_like(q, device=q.device), noise_mode='const', force_fp32=True)

            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            for i in tqdm(range(hyperparameters.max_pti_steps)):
                generated_deg_images = self.forward(w_pivot, q)
                generated_sharp_images = self.forward(w_pivot, torch.zeros_like(q, device=q.device))
                loss, l2_loss_val, loss_lpips, loss_percep = self.calc_loss(
                    generated_sharp_images,
                    org_sharp_images,
                    generated_deg_images,
                    org_deg_images,
                    real_images_batch,
                    image_name,
                    self.G,
                    use_ball_holder,
                    w_pivot
                )

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

            end = time.time()
            if self.image_counter >= 2:
                runtimes.append(end - start)

            if self.gen_degraded:
                with torch.no_grad():
                    degraded_image, _ = self.G.synthesis(w_pivot, q, noise_mode='const', force_fp32=True)

            if self.gen_degraded:
                degraded_image = (degraded_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
                resized_image = Image.fromarray(degraded_image, mode='RGB')
                save_path = os.path.join(self.save_dir, "degraded_gen", image_name + ".png")
                dir_name = os.path.dirname(save_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                resized_image.save(save_path)

            sharp_image, _ = self.G.synthesis(w_pivot, torch.zeros_like(q, device=q.device), noise_mode='const', force_fp32=True)
            sharp_image = (sharp_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
            resized_image = Image.fromarray(sharp_image, mode='RGB')
            save_path = os.path.join(self.save_dir, "sharp_gen", image_name + ".png")
            dir_name = os.path.dirname(save_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            resized_image.save(save_path)

            model_path = f'{self.save_dir}/models/{image_name}.pt'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.G, model_path)

        print(np.mean(runtimes), np.std(runtimes))
