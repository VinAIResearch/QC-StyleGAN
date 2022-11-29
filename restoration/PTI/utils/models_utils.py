import pickle

import torch
from configs import global_config, paths_config


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type):
    new_G_path = f"{paths_config.checkpoints_dir}/model_{run_id}_{type}.pt"
    with open(new_G_path, "rb") as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_old_G():
    with open(paths_config.stylegan2_ada_ffhq, "rb") as f:
        old_G = pickle.load(f)["G_ema"].to(global_config.device).eval()
        old_G = old_G.float()
    return old_G


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
