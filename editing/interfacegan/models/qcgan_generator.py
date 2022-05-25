import numpy as np

import torch

from . import model_settings
from .base_generator import BaseGenerator

import dnnlib.util
import legacy


__all__ = ["QCGANGenerator"]
seeds = iter([27,12,19,99,9318,10,100,1000,9,999])


class QCGANGenerator(BaseGenerator):
    def __init__(self, model_name, logger=None):
        self.truncation_psi = model_settings.STYLEGAN_TRUNCATION_PSI
        self.truncation_cutoff = model_settings.STYLEGAN_TRUNCATION_CUTOFF

        super(QCGANGenerator, self).__init__(model_name, logger)
        assert self.gan_type == "qcgan"

    def build(self):
        network_pkl = self.model_path
        print(self.model_path)
        if network_pkl.endswith('.pt'):
            # with dnnlib.util.open_url(network_pkl) as f:
            #     tmp = legacy.load_network_pkl(f)
            # self.model = Generator(**tmp["G"].init_kwargs)
            self.model = torch.load(network_pkl)
        else:
            with dnnlib.util.open_url(network_pkl) as f:
                self.model = legacy.load_network_pkl(f)["G_ema"].to(self.run_device)  # type: ignore
                self.logger.info(f"Loading pytorch model from `{network_pkl}`.")

    def load(self):
        pass

    def sample(self, num, latent_space_type="Z"):
        latent_space_type = latent_space_type.upper()
        if latent_space_type == "Z":
            latent_codes = np.random.RandomState(next(seeds)).randn(num, self.model.z_dim)
        elif latent_space_type == "W":
            latent_codes = np.random.randn(num, self.w_space_dim)
        else:
            raise NotImplementedError(f"Unrecognized {latent_space_type}")

        return latent_codes

    def preprocess(self, latent_codes, latent_space_type="Z"):
        return latent_codes

    def postprocess(self, images):
        return images

    def easy_sample(self, num, latent_space_type="Z"):
        return self.preprocess(self.sample(num, latent_space_type), latent_space_type)

    def synthesize(self, latent_codes, q=None, latent_space_type="Z", generate_style=False, generate_image=True):
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError("Latent codes should be with type `numpy.ndarray`!")

        results = {}

        latent_space_type = latent_space_type.upper()
        latent_codes_shape = latent_codes.shape

        if latent_space_type == "Z":
            if not (
                len(latent_codes_shape) == 2
                and latent_codes_shape[0] <= self.batch_size
                and latent_codes_shape[1] == self.latent_space_dim
            ):
                raise ValueError(
                    f"Latent_codes should be with shape [batch_size, "
                    f"latent_space_dim], where `batch_size` no larger "
                    f"than {self.batch_size}, and `latent_space_dim` "
                    f"equal to {self.latent_space_dim}!\n"
                    f"But {latent_codes_shape} received!"
                )

            z = torch.from_numpy(latent_codes).to(self.run_device)
            if q is None:
                q = torch.from_numpy(np.zeros([1, self.model.q_dim])).to(self.run_device)
            else:
                q = torch.from_numpy(q).to(self.run_device)
            label = torch.zeros([z.shape[0], self.model.c_dim]).to(self.run_device)
            ws = self.model.mapping(z, label)
            results['z'] = latent_codes
            results["w"] = ws.detach().cpu().numpy()
        elif latent_space_type == "W":
            if not (
                len(latent_codes_shape) == 2
                and latent_codes_shape[0] <= self.batch_size
                and latent_codes_shape[1] == self.w_space_dim
            ):
                raise ValueError(
                    f"Latent_codes should be with shape [batch_size, "
                    f"w_space_dim], where `batch_size` no larger than "
                    f"{self.batch_size}, and `w_space_dim` equal to "
                    f"{self.w_space_dim}!\n"
                    f"But {latent_codes_shape} received!"
                )
            if q is None:
                q = torch.from_numpy(np.zeros([1, self.model.q_dim])).to(self.run_device)
            else:
                q = torch.from_numpy(q).to(self.run_device)
            ws = torch.from_numpy(latent_codes).type(torch.FloatTensor).unsqueeze(1).repeat([1, self.model.num_ws, 1])
            ws = ws.to(self.run_device)
            # wps = self.model.truncation(ws)
            results["w"] = latent_codes
            # results["wp"] = self.get_value(wps)
        else:
            raise NotImplementedError(f"Unrecognized {latent_space_type}")

        if generate_image:
            img, _ = self.model.synthesis(ws, q, noise_mode='const')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            if self.channel_order == "BGR":
                img = img[:, :, :, ::-1]
            results["image"] = self.get_value(img)

        return results
