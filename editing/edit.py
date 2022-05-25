"""Edits latent codes with respect to given boundary.

Basically, this file takes latent codes and a semantic boundary as inputs, and
then shows how the image synthesis will change if the latent codes is moved
towards the given boundary.

NOTE: If you want to use W or W+ space of StyleGAN, please do not randomly
sample the latent code, since neither W nor W+ space is subject to Gaussian
distribution. Instead, please use `generate_data.py` to get the latent vectors
from W or W+ space first, and then use `--input_latent_codes_path` option to
pass in the latent vectors.
"""

import argparse
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm

from interfacegan.models.model_settings import MODEL_POOL
from interfacegan.models.qcgan_generator import QCGANGenerator
import glob
import os.path as osp
from interfacegan.utils.logger import setup_logger
from interfacegan.utils.manipulator import linear_interpolate


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description="Edit image synthesis with given semantic boundary.")
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Directory to save the output results. (required)"
    )
    parser.add_argument(
        "-b", "--boundary_path", type=str, required=True, help="Path to the semantic boundary. (required)"
    )
    parser.add_argument(
        "-i",
        "--input_latent_codes",
        type=str,
        default="",
        help="If specified, will load latent codes from given " "path instead of randomly sampling. (optional)",
    )
    parser.add_argument(
        "-m",
        "--models_path",
        type=str,
        default="",
        help="If specified, will load latent codes from given " "path instead of randomly sampling. (optional)",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1,
        help="Number of images for editing. This field will be "
        "ignored if `input_latent_codes_path` is specified. "
        "(default: 1)",
    )
    parser.add_argument(
        "-s",
        "--latent_space_type",
        type=str,
        default="z",
        choices=["z", "Z", "w", "W", "wp", "wP", "Wp", "WP"],
        help="Latent space used in Style GAN. (default: `Z`)",
    )
    parser.add_argument(
        "--start_distance",
        type=float,
        default=-3.0,
        help="Start point for manipulation in latent space. " "(default: -3.0)",
    )
    parser.add_argument(
        "--end_distance", type=float, default=3.0, help="End point for manipulation in latent space. " "(default: 3.0)"
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of steps for image editing. (default: 10)")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    logger = setup_logger(args.output_dir, logger_name="generate_data")

    logger.info("Preparing boundary.")
    if not os.path.isfile(args.boundary_path):
        raise ValueError(f"Boundary `{args.boundary_path}` does not exist!")
    boundary = np.load(args.boundary_path)
    np.save(os.path.join(args.output_dir, "boundary.npy"), boundary)

    logger.info("Preparing latent codes.")

    print(args.models_path)
    latent_paths = sorted(glob.glob(osp.join(args.input_latent_codes, '*.pt')))[:1]
    model_paths = sorted(glob.glob(osp.join(args.models_path, '*.pt')))[:1]

    assert len(latent_paths) == len(model_paths)
    total_num = len(model_paths)

    print(f'Number of latent codes: {total_num}')
    print(f'Number of models: {len(model_paths)}')

    for sample_id in tqdm(range(total_num), leave=False):
        latent = torch.load(latent_paths[sample_id]).unsqueeze(0).cpu().numpy()
        w, q = latent[:, :512], latent[:, 512:]

        logger.info("Initializing generator.")
        args.model_name = 'stylegan2_ada_ffhq'
        MODEL_POOL['stylegan2_ada_ffhq']['model_path'] = model_paths[sample_id]
        model = QCGANGenerator(args.model_name, logger)
        kwargs = {"latent_space_type": args.latent_space_type}

        interpolations = linear_interpolate(
            w,
            boundary,
            start_distance=args.start_distance,
            end_distance=args.end_distance,
            steps=args.steps,
        )
        interpolation_id = 0
        interpolation_sharp_id = 0
        for interpolations_batch in model.get_batch_inputs(interpolations):
            outputs = model.easy_synthesize(
                interpolations_batch, q=q, **kwargs
            )
            outputs_sharp = model.easy_synthesize(
                interpolations_batch, **kwargs
            )
            for image in outputs["image"]:
                save_path = os.path.join(args.output_dir, f"{sample_id:03d}_{interpolation_id:03d}.jpg")
                cv2.imwrite(save_path, image[:, :, ::-1])
                interpolation_id += 1
            for image in outputs_sharp["image"]:
                save_path = os.path.join(args.output_dir, f"sharp_{sample_id:03d}_{interpolation_sharp_id:03d}.jpg")
                cv2.imwrite(save_path, image[:, :, ::-1])
                interpolation_sharp_id += 1
        assert interpolation_id == args.steps
        logger.debug(f"  Finished sample {sample_id:3d}.")
    logger.info(f"Successfully edited {total_num} samples.")


if __name__ == "__main__":
    main()
