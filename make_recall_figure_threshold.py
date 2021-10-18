import importlib
import os
import pickle
import argparse
import statistics
import torch

import scipy.interpolate
import numpy as np
import matplotlib.font_manager
import matplotlib.pyplot as plt

from utils.tester import Tester
from datasets.datasets import get_dataloader
from tqdm import tqdm
from utils.utils import get_data_in_np_array, transform_pc, get_num_key_points, get_output_file_name


# define visualization parameters
LINE_WIDTH = 2
FONT_SIZE = 18
FIG_SIZE = 4
MARKER_SIZE = 7

linestyle_str = ['solid', 'dotted', 'dashed', 'dashdot']*10
markers = [".", "o", "*", "+", "x"]*10

textcolor = 'k'
bg_color = 'w'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.color": textcolor,
    "axes.labelcolor": textcolor,
    "axes.edgecolor": textcolor,
    "xtick.color": textcolor,
    "ytick.color": textcolor,
    "lines.color": textcolor,
    "legend.facecolor": bg_color
})


name_mapping = {
    "icp_pt2pl_o3d": "ICP (pt2pl)",
    "icp_pt2pt_o3d": "ICP (pt2pt)",
    "fpfh_ransac": "FPFH (RANSAC)",
    "fpfh_teaser": "FPFH (TEASER++)",
    "fcgf_ransac": "FCGF (RANSAC)",
    "fcgf_teaser": "FCGF (TEASER++)",
    "d3feat_ransac": "D3Feat (RANSAC)",
    "d3feat_teaser": "D3Feat (TEASER++)",
    "use_ini_pose": "ini"}


dict_groups = {
    "deep feature point": [
        "fcgf_ransac",
        "fcgf_teaser",
        "d3feat_ransac",
        "d3feat_teaser",
    ],
    "raw point": ["icp_pt2pt_o3d",
                  "icp_pt2pl_o3d"],
    "feature point": ["fpfh_teaser",
                      "fpfh_ransac"],
}

color_groups = {"deep feature point": (0.41, 0.69, 1.0),
                "raw point": (0.46, 0.77, 0.45),
                "feature point": (0.88, 0.44, 0.44)}


list_anchor_map_size = [1, 3, 10, 30]
list_success_th_trans = np.arange(0.1, 5.1, 0.5).tolist()


def get_color(method):
    for i, key in enumerate(dict_groups.keys()):
        if method in dict_groups[key]:
            return color_groups[key]


def compute_map_area(dataloader):
    print("Compute total map area...")
    area_sum = 0
    for idx, data in tqdm(enumerate(dataloader)):

        source, target, T_gt, _ = get_data_in_np_array(data)

        x_min, x_max = target.points[:, 0].min(), target.points[:, 0].max()
        y_min, y_max = target.points[:, 1].min(), target.points[:, 1].max()

        x_min, x_max, y_min, y_max = int(x_min), int(
            x_max), int(y_min), int(y_max)
        area = np.zeros((x_max-x_min+1, y_max-y_min+1))

        area[target.points[:, 0].astype(
            np.int)-x_min, target.points[:, 1].astype(np.int)-y_min] = 1

        area_sum += area.sum()

    print("area_sum = ", area_sum)
    return area_sum


def compute_DS(list_data_size, list_success_rate, anchor, limit_range=False):

    f = scipy.interpolate.interp1d(
        list_data_size, list_success_rate, fill_value="extrapolate")

    boundary = 2
    if anchor > list_data_size[-1]+boundary:  # no valid data for interpolation
        return None

    y = f(anchor)
    if limit_range:
        y[y > 1] = 1
        y[y < 0] = 0

    return y.item()


def main(args):

    cfg = importlib.import_module(args.path_cfg)
    tester = Tester(cfg)

    dataloader = get_dataloader(cfg)
    area_sum = compute_map_area(dataloader)

    # initialize plots
    num_plots = len(list_anchor_map_size)

    fig, ax_recall_all = plt.subplots(1, num_plots,  figsize=(
        num_plots*FIG_SIZE+0.1, FIG_SIZE*0.9), sharey=False)
    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.05, right=0.85)

    fig.patch.set_facecolor(bg_color)

    fig_te, ax_te_all = plt.subplots(1, len(cfg.list_num_key_points),  figsize=(
        len(cfg.list_num_key_points)*FIG_SIZE+0.1, FIG_SIZE*0.9), sharey=False)
    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.05, right=0.85)

    fig_te.patch.set_facecolor(bg_color)

    for i, ax_recall in enumerate(ax_recall_all):
        if i == 0:
            ax_recall.set_ylabel("success rate", fontsize=FONT_SIZE)
        ax_recall.set_xlabel(r"success th($m$)", fontsize=FONT_SIZE)
        plt.setp(ax_recall, yticks=np.arange(0, 1, step=0.2))
        ax_recall.set_ylim([0, 1])
        ax_recall.set_xlim([0, 5])
        ax_recall.set_title(rf"map size = {list_anchor_map_size[i]} $bytes/m^{2}$", fontsize=FONT_SIZE)
        ax_recall.tick_params(labelsize=12)

        ax_recall.set_facecolor(bg_color)

    for i, ax_te in enumerate(ax_te_all):
        if i == 0:
            ax_te.set_ylabel("density", fontsize=FONT_SIZE)
        ax_te.set_xlabel(r"x error($m$)", fontsize=FONT_SIZE)
        plt.setp(ax_te, yticks=np.arange(0, 500, step=100))
        ax_te.set_ylim([0, 1])
        ax_te.set_xlim([-11, 11])
        ax_te.set_title(rf"pt num = {cfg.list_num_key_points[i]}", fontsize=FONT_SIZE)
        ax_te.tick_params(labelsize=12)

        ax_te.set_facecolor(bg_color)

    print("Start evaluation...")

    for ind_map_size, map_size_anchor in enumerate(list_anchor_map_size):
        for i, method in enumerate(cfg.list_methods):
            list_components = cfg.list_num_key_points

            if 'ransac' in method or 'teaser' in method:
                feat_type = method.split('_')[0]
            else:
                feat_type = method

            list_recall_inter = []
            for ind_suc, suc_th in enumerate(list_success_th_trans):
                list_recall = []
                list_prec_t = []
                list_data_size = []

                for ind_comp, num_raw_points in enumerate(list_components):

                    filename_output = get_output_file_name(
                        args, cfg, num_raw_points, method, feat_type)

                    num_key_points = get_num_key_points(
                        cfg.dict_methods[feat_type]['data_size'], num_raw_points)

                    path_results = os.path.join(cfg.path_output, f"{filename_output}.results")

                    if os.path.exists(path_results):
                        with open(path_results, "rb") as f:
                            results = pickle.load(f)
                    else:
                        raise ValueError(f"Missing : {path_results}")

                    list_suc = [x < suc_th for x in results['list_rte_all']]

                    list_data_size.append(
                        sum(results['list_data_size'])/area_sum)

                    list_recall.append(statistics.mean(list_suc))

                    # visualize the x translation error distribution with respect to key point num
                    # this doesn't depend on success threshold and map size
                    if ind_suc == 0 and ind_map_size == 0:
                        x_error = [est[0, 3]-gt[0, 3] for est,
                                   gt in zip(results['list_T_est'], results['list_T_gt'])]

                        bins = np.linspace(-11, 11, 50)
                        y_vals, _ = np.histogram(
                            x_error, bins=bins, density=True)

                        ax_te_all[ind_comp].plot(bins[0:-1]+(bins[1]-bins[0])/2,
                                                 y_vals,
                                                 linewidth=LINE_WIDTH/2,
                                                 label=f"{name_mapping[method]}", color=get_color
                                                 (method),
                                                 linestyle=linestyle_str[i])
                # apply interpolation
                list_recall_inter.append(compute_DS(
                    list_data_size, list_recall, map_size_anchor, limit_range=False))

                print(method, "(data size)", list_data_size,
                      len(results['list_data_size']))
                print(method, "(recall)", list_recall,
                      len(results['list_data_size']))

            ax_recall_all[ind_map_size].plot(list_success_th_trans, list_recall_inter, linewidth=LINE_WIDTH,
                                             label=f"{name_mapping[method]}", color=get_color
                                             (method),
                                             markersize=MARKER_SIZE, linestyle=linestyle_str[i],
                                             marker=markers[i])

    ax_recall_all[-1].legend(bbox_to_anchor=(1.05, 1),
                             loc='upper left', fontsize=12)

    ax_te_all[-1].legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', fontsize=12)

    path_output = os.path.join(cfg.path_output, f"output_recall_{cfg.dataset_name}.jpg")
    fig.savefig(path_output)
    path_output = os.path.join(cfg.path_output, f"output_x_error_{cfg.dataset_name}.jpg")
    fig_te.savefig(path_output)

    print(f"Saved results to {cfg.path_output}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--flag",
                        type=str,
                        default="",
                        help="label which experiment is running")
    parser.add_argument("--path_cfg",
                        type=str,
                        help="path to cfg file (without \".py\")",
                        required=True)
    parser.add_argument("--name_log", type=str, default="output")
    args = parser.parse_args()
    print(args)
    main(args)
