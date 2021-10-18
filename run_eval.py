
import pickle
import importlib
import os
import argparse
import numpy as np

from datasets.datasets import get_dataloader
from tqdm import tqdm
from utils.tester import Tester
from utils.utils import get_data_in_np_array, transform_pc, get_num_key_points,\
    get_output_file_name


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_cfg', type=str,
                        help='path to cfg file (without \'.py\')', required=True)
    parser.add_argument('--name_log', type=str, default='output')
    args = parser.parse_args()
    return args


def compute_errors(T_est, T_gt):
    rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]).item()
    rre = (np.trace(T_est[:3, :3].T @ T_gt[:3, :3]) - 1) / 2
    rre = np.arccos(np.clip(rre, -1, 1))

    return rte, rre


if __name__ == '__main__':

    args = get_args()
    cfg = importlib.import_module(args.path_cfg)

    print('*********************************')
    print(f"Use configs from {args.path_cfg}")
    print(f"Dataset name = {cfg.dataset_name}")
    print(f"Testing {len(cfg.list_methods)} methods:", cfg.list_methods)
    print(f"Testing cases:", cfg.list_num_key_points)
    print(f"Additional name tag = {args.name_log}")
    print('*********************************')

    dataloader = get_dataloader(cfg)
    tester = Tester(cfg)

    # Loop though all the methods defined in cfg
    for method in cfg.list_methods:
        if 'ransac' in method or 'teaser' in method:
            feat_type = method.split('_')[0]
        else:
            feat_type = method

        if cfg.dict_methods[feat_type]["precomputed"]:
            print(f"Load precomputed features for {method}..")

            with open(os.path.join(cfg.dict_methods[feat_type]["path_precomputed"],
                                   f"{feat_type}.results.pkl.{cfg.dataset_name}"), "rb") as f:
                data_precomputed = pickle.load(f)

        for num_raw_points in cfg.list_num_key_points:

            filename_output = get_output_file_name(
                args, cfg, num_raw_points, method, feat_type)
            num_key_points = get_num_key_points(
                cfg.dict_methods[feat_type]['data_size'], num_raw_points)

            path_results = os.path.join(cfg.path_output, f"{filename_output}.results")
            print(f"Result path = {path_results}")

            if os.path.exists(path_results):
                print(
                    'Results exist! Load results from %s and only run evaluation.' % path_results)
                with open(path_results, 'rb') as f:
                    dict_result = pickle.load(f)
            else:
                print("Results don't exist. Generate results...")

                list_rte_success, list_rre_success = [], []
                list_rte_all, list_rre_all = [], []
                list_success = []
                list_data_size = []
                list_T_est = []
                list_T_gt = []

                for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):

                    source, target, T_gt, T_init = get_data_in_np_array(data)

                    if method == 'use_ini_pose':
                        # Evaluation the initial condition
                        T_est = np.eye(4)
                    else:
                        # Feature-based robust registration
                        if 'ransac' in method or 'teaser' in method:

                            if cfg.dict_methods[feat_type]['precomputed']:
                                # Precomputed deep features: d3feat or fcgf
                                source_pts = data_precomputed[idx]['pts_source']
                                target_pts = data_precomputed[idx]['pts_target']
                                source_feat = data_precomputed[idx]['feat_source']
                                target_feat = data_precomputed[idx]['feat_target']

                                # pre-extracted local maps are in map coordinates
                                target_pts = transform_pc(
                                    target_pts, np.linalg.inv(T_init))

                                if 'score_source' in data_precomputed[idx].keys():
                                    ind_sort = np.argsort(
                                        data_precomputed[idx]['score_target'][:, 0])
                                    num_key_points = min(num_key_points,
                                                         target_pts.shape[0])
                                    # not downsampling the source
                                    target_pts = target_pts[ind_sort[-num_key_points:]]
                                    target_feat = target_feat[ind_sort[-num_key_points:]]

                                else:
                                    # random downsampling when no scores were provided
                                    target_pts = target_pts[:num_key_points]
                                    target_feat = target_feat[:num_key_points]

                            else:
                                # non-deep feature point based method
                                # not downsampling the source
                                source_pts, source_feat = tester.downsample_and_extract_fpfh(
                                    source.points, cfg.voxel_size,
                                    None, T_init)

                                target_pts, target_feat = tester.downsample_and_extract_fpfh(
                                    target.points, cfg.voxel_size, num_key_points,
                                    T_init)

                            T_est = tester.get_registration_function(method)(
                                source_pts, target_pts, source_feat, target_feat,
                                num_key_points)

                        else:
                            source_pts = source.points
                            target_pts = target.points
                            # We will downsample the point cloud in the function after computing normals
                            T_est = tester.get_registration_function(method)(
                                source_pts, target_pts, num_key_points)

                    rte, rre = compute_errors(T_est, T_gt)
                    list_rte_all.append(rte)
                    list_rre_all.append(rre * 180 / np.pi)
                    list_T_est.append(T_est)
                    list_T_gt.append(T_gt)

                    if rte < cfg.success_th_trans:
                        list_rte_success.append(rte)

                    if not np.isnan(rre) and rre < np.pi / 180 * cfg.success_th_rot:
                        list_rre_success.append(rre * 180 / np.pi)

                    if (rte < cfg.success_th_trans) and (not np.isnan(rre)) and (
                            rre < np.pi / 180 * cfg.success_th_rot):
                        list_success.append(1)
                    else:
                        list_success.append(0)

                    list_data_size.append(
                        tester.get_data_size(
                            min(target.points.shape[0], num_key_points), method))

                dict_result = {
                    'list_rte_all': list_rte_all,
                    'list_rre_all': list_rre_all,
                    'list_rte_success': list_rte_success,
                    'list_rre_success': list_rre_success,
                    'list_success': list_success,
                    'list_data_size': list_data_size,
                    'list_T_est': list_T_est,
                    'list_T_gt': list_T_gt
                }

                print(f"Dump results to {path_results}")
                with open(path_results, 'wb') as f:
                    pickle.dump(dict_result, f)
