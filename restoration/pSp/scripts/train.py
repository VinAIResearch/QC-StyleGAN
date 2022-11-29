"""
This file runs the main training/val loop
"""
import json
import os
import pprint
import sys

from options.train_options import TrainOptions


sys.path.append(".")
sys.path.append("..")


def main():
    opts = TrainOptions().parse()
    if os.path.exists(opts.exp_dir):
        raise Exception("Oops... {} already exists".format(opts.exp_dir))
    os.makedirs(opts.exp_dir)

    if opts.coach == "q_encoder":
        from training.q_encoder_coach import Coach
    elif opts.coach == "alternative":
        from training.alternative_coach import Coach
    elif opts.coach == "w_encoder":
        from training.w_encoder_coach import Coach

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, "opt.json"), "w") as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    coach = Coach(opts)
    coach.train()


if __name__ == "__main__":
    main()
