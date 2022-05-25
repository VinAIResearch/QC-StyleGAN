import os
import os.path as osp
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp


def run():
    test_opts = TestOptions().parse()

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    is_cars = 'cars_' in opts.dataset_type
    out_path = opts.out_path

    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []

    all_latents = None
    os.makedirs(osp.join(out_path, 'inverted'), exist_ok=True)
    os.makedirs(osp.join(out_path, 'org'), exist_ok=True)
    os.makedirs(osp.join(out_path, 'latent'), exist_ok=True)

    total = 0
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            img_batch, latent_batch = run_on_batch(input_cuda, net, opts)

            if all_latents is None:
                all_latents = latent_batch
            else:
                all_latents = torch.cat((all_latents, latent_batch))

            for i in range(opts.test_batch_size):
                img = tensor2im(img_batch[i])
                latent = latent_batch[i]
                gt = tensor2im(input_batch[i])

                Image.fromarray(np.array(img)).save(f'{out_path}/inverted/{total:04d}.png')
                Image.fromarray(np.array(gt)).save(f'{out_path}/org/{total:04d}.png')
                torch.save(latent, f'{out_path}/latent/{total:04d}.pt')
                total += 1

            toc = time.time()
            global_time.append(toc - tic)

    torch.save(all_latents, 'inversed_latents.pt')

    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)


def run_on_batch(inputs, net, opts):
    result_batch = net(inputs, return_latents=True)
    return result_batch


if __name__ == '__main__':
    run()
