
import json
import os
from pathlib import Path
from utils.utils import make_folder_if_not_exist

success_th_trans = 5  # meter
success_th_rot = 5  # degree
noise_trans = 10  # meter
noise_rot = 10  # degree
# voxel_size = 0.3  # meter

list_methods = ['icp_pt2pt_o3d',
                'icp_pt2pl_o3d',
                'fpfh_teaser', 'fpfh_ransac',
                'd3feat_teaser', 'd3feat_ransac',
                'fcgf_teaser', 'fcgf_ransac',
                'use_ini_pose']

dataset_name = 'KITTI'  # 'KITTI'or 'Argoverse_Tracking'

path = os.path.dirname(Path(__file__))
with open(os.path.join(path, 'methods.json')) as json_data_file:
    dict_methods = json.load(json_data_file)

path_preprocessed_data = './data'

# number of 3-float raw points
list_num_key_points = [100, 300, 1000, 3000, 5000, 10000]

path_data = os.path.join(path_preprocessed_data, 'data_%s' % (dataset_name))
path_output = 'output_eval_%s' % (dataset_name)

exp_name = f"noise_level_{noise_trans}_{noise_rot}"

make_folder_if_not_exist(path_output)
