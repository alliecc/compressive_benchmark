# code partially from D3feat repo
import numpy as np
import pickle
import os
import csv
import open3d as o3d


def make_open3d_pc(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_open3d_feat(features):
    feat = o3d.pipelines.registration.Feature()
    feat.resize(features.shape[1], features.shape[0])
    feat.data = features
    return feat


def get_output_file_name(args, cfg, num_raw_points, method, feat_type):

    filename_output = f"{args.name_log}_{cfg.exp_name}_{num_raw_points}_{method}"

    for key in cfg.dict_methods[feat_type].keys():
        if key != 'path_precomputed':
            filename_output += f"_{cfg.dict_methods[feat_type][key]}"

    return filename_output


def get_num_key_points(data_size, num_raw_points):
    num_key_points = int(
        num_raw_points / (data_size / 3))
    return num_key_points


def transform_pc(pc, T):
    return ((T[0:3, 0:3] @ pc.transpose()).transpose() + T[0:3, 3])


def make_folder_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_data_in_np_array(data):
    # assume batch size = 1
    source = data[0]['source'].from_tensor_to_np_array()
    target = data[0]['target'].from_tensor_to_np_array()

    return source, target, data[0]['T_gt'].numpy(), data[0]['T_init'].numpy()
