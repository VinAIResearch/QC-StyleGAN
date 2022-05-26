# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class KALoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X, Y):
        X_ = X.view(X.size(0), -1)
        Y_ = Y.view(Y.size(0), -1)
        assert X_.shape[0] == Y_.shape[
            0], f'X_ and Y_ must have the same shape on dim 0, but got {X_.shape[0]} for X_ and {Y_.shape[0]} for Y_.'
        X_vec = X_ @ X_.T
        Y_vec = Y_ @ Y_.T
        ret = (X_vec * Y_vec).sum() / ((X_vec**2).sum() * (Y_vec**2).sum())**0.5
        return ret

class KDLoss(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        assert name in ['ka', 'l2'], f"Not supported knowledge distillation {name} Loss!"
        if name == "l2":
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = KALoss()
      
    def forward(self, yhat, y):
        v = self.loss(yhat, y)
        return v

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_degraded_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, D_degraded, q_dim, Gteacher_synthesis, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.Gteacher_synthesis = Gteacher_synthesis
        self.D = D
        self.D_degraded = D_degraded
        self.q_dim = q_dim
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.pl_degraded_mean = torch.zeros([], device=self.device)
        self.kdloss = KDLoss('l2')

    def run_G(self, z, c, q, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img, _ = self.G_synthesis(ws, q)
        return img, ws
    def run_G_mapping(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        return ws

    def run_G_synthesis(self, ws, q, sync):
        with misc.ddp_sync(self.G_synthesis, sync):
            img, x_c = self.G_synthesis(ws, q)
        return img, x_c

    def run_Gteacher_synthesis(self, ws, sync):
        with misc.ddp_sync(self.Gteacher_synthesis, sync):
            img, x_c = self.Gteacher_synthesis(ws)
        return img, x_c

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def run_D_degraded(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D_degraded, sync):
            logits = self.D_degraded(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_degraded_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth',\
            'D_degradedmain', 'D_degradedreg', 'D_degradedboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        do_D_degraded_main = (phase in ['D_degradedmain', 'D_degradedboth'])
        do_D_degraded_r1   = (phase in ['D_degradedreg', 'D_degradedboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                latent_shape = list(gen_z.shape)
                latent_shape[-1] = self.q_dim
                q = torch.zeros(latent_shape, device=self.device)
                q = q.detach().requires_grad_(True)
                _gen_ws = self.run_G_mapping(gen_z, gen_c, sync=(sync and not do_Gpl))
                gen_teacher_img, gen_teacher_features = self.run_Gteacher_synthesis(_gen_ws, sync=(sync and not do_Gpl))
                gen_img, gen_features = self.run_G_synthesis(_gen_ws, q, sync=(sync and not do_Gpl))

                q = torch.randn(latent_shape, device=self.device)
                q = q.detach().requires_grad_(True)

                gen_degraded_img, _ = self.run_G_synthesis(_gen_ws, q, sync=(sync and not do_Gpl))

                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                gen_degraded_logits = self.run_D_degraded(gen_degraded_img, gen_c, sync=False)

                kd_logits = 0
                for genhat, gen in zip(gen_features, gen_teacher_features):
                    kd_logits += self.kdloss(genhat, gen)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/scores/fake_degraded', gen_degraded_logits)
                training_stats.report('Loss/signs/fake_degraded', gen_degraded_logits.sign())
                training_stats.report('Loss/scores/content', kd_logits)
                training_stats.report('Loss/signs/content', kd_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) + torch.nn.functional.softplus(-gen_degraded_logits) + kd_logits * 3 # -log(sigmoid(gen_logits))

                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                latent_shape = list(gen_z.shape)
                latent_shape[-1] = self.q_dim
                latent_shape[0]   = batch_size
                q = torch.zeros(latent_shape, device=self.device)
                q = q.detach().requires_grad_(True)
                gen_ws = self.run_G_mapping(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                gen_img, _ = self.run_G_synthesis(gen_ws, q, sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws, q], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)

                q = torch.randn(latent_shape, device=self.device)
                q = q.detach().requires_grad_(True)
                gen_degraded_img, _ = self.run_G_synthesis(gen_ws, q, sync=sync)

                pl_degraded_noise = torch.randn_like(gen_degraded_img) / np.sqrt(gen_degraded_img.shape[2] * gen_degraded_img.shape[3])
                with torch.autograd.profiler.record_function('pl_degraded_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_degraded_grads = torch.autograd.grad(outputs=[(gen_degraded_img * pl_degraded_noise).sum()], inputs=[gen_ws, q], create_graph=True, only_inputs=True)[0]
                pl_degraded_lengths = pl_degraded_grads.square().sum(2).mean(1).sqrt()
                pl_degraded_mean = self.pl_degraded_mean.lerp(pl_degraded_lengths.mean(), self.pl_decay)
                self.pl_degraded_mean.copy_(pl_degraded_mean.detach())
                pl_degraded_penalty = (pl_degraded_lengths - pl_degraded_mean).square()
                training_stats.report('Loss/pl_penalty_degraded', pl_degraded_penalty)
                loss_degraded_Gpl = pl_degraded_penalty * self.pl_weight
                training_stats.report('Loss/G/reg_degraded', loss_degraded_Gpl)

            with torch.autograd.profiler.record_function('Gpl_backward'):
                _Gpl = (gen_img[:, 0, 0, 0] * 0 + loss_Gpl) + (gen_degraded_img[:, 0, 0, 0] * 0 + loss_degraded_Gpl)
                _Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                latent_shape = list(gen_z.shape)
                latent_shape[-1] = self.q_dim
                q = torch.zeros(latent_shape, device=self.device)
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, q, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated degraded images.
        loss_Dgen_degraded = 0
        if do_D_degraded_main:
            with torch.autograd.profiler.record_function('D_degraded_gen_forward'):
                latent_shape = list(gen_z.shape)
                latent_shape[-1] = self.q_dim
                q = torch.randn(latent_shape, device=self.device)
                gen_degraded_img, _gen_degraded_ws = self.run_G(gen_z, gen_c, q, sync=False)
                gen_degraded_logits = self.run_D_degraded(gen_degraded_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake_degraded', gen_degraded_logits)
                training_stats.report('Loss/signs/fake_degraded', gen_degraded_logits.sign())
                loss_Dgen_degraded = torch.nn.functional.softplus(gen_degraded_logits) * 2 # -log(1 - sigmoid(gen_logits))

            with torch.autograd.profiler.record_function('D_degraded_gen_backward'):
                loss_Dgen_degraded.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # Dmain: Maximize logits for real degraded images.
        # Dr1: Apply R1 regularization.
        if do_D_degraded_main or do_D_degraded_r1:
            name = 'Dreal_Dr1_degraded' if do_D_degraded_main and do_D_degraded_r1 else 'Dreal_degraded' if do_D_degraded_main else 'Dr1_degraded'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_degraded_img_tmp = real_degraded_img.detach().requires_grad_(do_D_degraded_r1)
                real_degraded_logits = self.run_D_degraded(real_degraded_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real_degraded', real_degraded_logits)
                training_stats.report('Loss/signs/real_degraded', real_degraded_logits.sign())

                loss_Dreal_degraded = 0
                if do_D_degraded_main:
                    loss_Dreal_degraded = torch.nn.functional.softplus(-real_degraded_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_degraded', loss_Dgen_degraded + loss_Dreal_degraded)

                loss_Dr1_degraded = 0
                if do_D_degraded_r1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_degraded_grads = torch.autograd.grad(outputs=[real_degraded_logits.sum()], inputs=[real_degraded_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_degraded_penalty = r1_degraded_grads.square().sum([1,2,3])
                    loss_Dr1_degraded = r1_degraded_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty_degraded', r1_degraded_penalty)
                    training_stats.report('Loss/D/reg_degraded', loss_Dr1_degraded)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_degraded_logits * 0 + loss_Dreal_degraded + loss_Dr1_degraded).mean().mul(gain).backward()

#----------------------------------------------------------------------------
