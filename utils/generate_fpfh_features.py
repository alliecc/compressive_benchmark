from utils.utils import make_folder_if_not_exist, error_vis, get_data_in_np_array
from datasets.datasets import PairwiseDataset, collate_fn
import matplotlib.pyplot as plt
from utils.tester import Tester
import argparse
import os
import logging
import importlib
import torch
import statistics
import pickle
import numpy as np


LINE_WIDTH = 3
FONT_SIZE = 12
FIG_SIZE = 4

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--flag",
                        type=str,
                        default="",
                        help="label which experiment is running")
    parser.add_argument("--path_cfg",
                        type=str,
                        help="path to cfg file (without \".py\")",
                        required=True)

    args = parser.parse_args()
    print(args)

    cfg = importlib.import_module(args.path_cfg)

    fig, (ax_trans, ax_rot, ax_suc) = plt.subplots(
        1, 3,  figsize=(FIG_SIZE*3, FIG_SIZE), sharey=False)

    make_folder_if_not_exist(cfg.path_output)

    dict_result_all = {}
    dataset = PairwiseDataset("test", cfg)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             collate_fn=collate_fn)

    tester = Tester(cfg)
    for num_key_points in cfg.list_num_key_points:
        filename_output = f"{args.name_log}_{cfg.exp_name}_{cfg.method}_{num_key_points}"

        logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(cfg.path_output,
                                                  f"{filename_output}.log"),
                            filemode='w',
                            format="")

        for idx, data in enumerate(dataloader):

            if (idx % float(len(dataloader) // 5)) == 0:
                print(f"{num_key_points} : {idx}/{len(dataloader)}")

            source, target, T_gt = get_data_in_np_array(data)
