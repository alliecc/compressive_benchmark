
# Benchmark tools for Compressive LiDAR-to-map registration

This repo contains the released version of code and datasets used for our IROS 2021 paper: "Map Compressibility Assessment for LiDAR Registration [[link]](https://www.cs.cmu.edu/~kaess/pub/Chang21iros.pdf). If you find the code useful for your work, please cite:

```bibtex
@inproceedings{Chang21iros,
   author = {M.-F. Chang and W. Dong and J.G. Mangelson and M. Kaess and S. Lucey},
   title = {Map Compressibility Assessment for {LiDAR} Registration},
   booktitle = {Proc. IEEE/RSJ Intl. Conf. on Intelligent Robots andSystems, IROS},
   address = {Prague, Czech Republic},
   month = sep,
   year = {2021}
}
```

# Environment Setup

The released codebase supports following methods:
1. Point-to-point ICP (from open3d)
2. Point-to-plane ICP (from open3d)
3. FPFH (with RANSAC from open3d or Teaser++)
4. FCGF (with RANSAC from open3d or Teaser++)
5. D3Feat (with RANSAC from open3d or Teaser++)

To run Teaser++, please also install from https://github.com/MIT-SPARK/TEASER-plusplus (python bindings required). One can build install the environment with the following conda command:

```shell script
conda create --name=benchmark  python=3.6  numpy open3d=0.12  tqdm pytorch cpuonly -c pytorch -c open3d-admin -c conda-forge 
conda activate benchmark
pip install pillow==6.0 #for visualization
```

# Datasets

The preprocessed data can be downloaded from [[link]](https://drive.google.com/drive/folders/1kfSmi1-ZUctKj_mBj2_FwxOhyyvp6TTb?usp=sharing).
The following data were provided:
1. Preprocessed KITTI scan/local map pairs
2. Preprocessed Argoverse Tracking scan/local map pairs
3. FCGF and D3Feat features
4. The ground truth poses

We haved preprocessed the results from FCGF and D3Feat into pickle files. The dataset is organized as source-target pairs. The source is the input LiDAR scan and the target is the cropped local map.

By default, we put the data in ./data folder. Please download the corresponding files from [[link]](https://drive.google.com/drive/folders/1kfSmi1-ZUctKj_mBj2_FwxOhyyvp6TTb?usp=sharing) and put/symlink it in ./data. The file structure is as follows:

```shell
./data
   ├─ data_Argoverse_Tracking
   │    ├─ test_dict_maps.pickle
   │    ├─ test_list_T_gt.pickle
   │    └─ test_samples.pickle
   │ 
   ├─ data_KITTI
   │    ├─ test_dict_maps.pickle
   │    ├─ test_list_T_gt.pickle
   │    └─ test_samples.pickle
   │ 
   ├─ deep
   │    ├─ d3feat.results.pkl.Argoverse_Tracking
   │    ├─ d3feat.results.pkl.KITTI
   │    ├─ fcgf.results.pkl.Argoverse_Tracking
   │    └─ fcgf.results.pkl.KITTI
----
```
# Usage

To run the code, simply use the following command and specify the config file name.:
```shell
python3 run_eval.py --path_cfg=configs.config
```

For trying out existing methods, first edit [config.py](https://github.com/alliecc/compressive_benchmark/blob/master/configs/config.py) to config the method list, the dataset name, and the local dataset path. 

For trying out new methods, please add the registration function to [tester.py](https://github.com/alliecc/compressive_benchmark/blob/master/utils/tester.py) and add the method configuration to [method.py](https://github.com/alliecc/compressive_benchmark/blob/master/configs/methods.json) and the parameters to method.json.

To visualize the resulting recall curves, please run
```shell
python3 make_recall_figure_threshold.py --path_cfg=configs.config
```
It will generate the recall plot and error density plot in ./output_eval_{dataset_name}. Here is an expected outout:

<img src="https://github.com/alliecc/compressive_benchmark/blob/master/images/output_recall_KITTI.jpg" width="800"> 

<img src="https://github.com/alliecc/compressive_benchmark/blob/master/images/output_x_error_KITTI.jpg" width="1000"> 

# Acknowledgement
This work was supported by the CMU Argo AI Center for Autonomous Vehicle Research. We also thank our labmates for the valuable suggestions to improve this paper.

# References
1. [Teaser++](https://github.com/MIT-SPARK/TEASER-plusplus)
2. [Open3d](http://www.open3d.org/)
3. [KITTI Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
4. [Argoverse 3D Tracking 1.1](https://www.argoverse.org/data.html)
5. [FCGF](https://github.com/chrischoy/FCGF)
6. [D3Feat](https://github.com/XuyangBai/D3Feat)
