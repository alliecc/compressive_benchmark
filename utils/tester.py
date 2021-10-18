import open3d as o3d
import numpy as np

from utils.utils import make_open3d_pc, make_open3d_feat
from utils.feat_match import find_knn_cpu, match_feats


class Tester():
    def __init__(self, cfg):
        '''
        param cfg: the config module from importlib
        '''
        self.cfg = cfg

    def get_data_size(self, pts_num, method):
        '''
        param pts_num: int for the number of points
        param method: str for the method name
        '''
        method = method.split(
            '_')[0] if 'teaser' in method or 'ransac' in method else method

        if self.cfg.dict_methods[method]['point_based']:
            # convert number of points into bytes
            return pts_num * self.cfg.dict_methods[method]['data_size'] * 4

        else:
            raise NotImplementedError(
                'Data size function for method {} is not implemented'.format
                (method))

    def get_registration_function(self, method):
        '''
        param method: str for the method name
        '''
        self.method = method

        if 'teaser' in method:
            return self.run_global_registration_teaser
        elif 'ransac' in method:
            return self.run_global_registration_ransac

        else:
            if method == 'icp_pt2pt_o3d':
                return self.run_registration_points_icp_pt2pt_o3d
            elif method == 'icp_pt2pl_o3d':
                return self.run_registration_points_icp_pt2pl_o3d
            else:
                raise NotImplementedError(
                    'Method {} is not implemented'.format(method))

    def downsample_and_extract_fpfh(self,
                                    pc,
                                    voxel_size,
                                    num_key_points=None, T_init=None):
        '''
        param pc:(N, 3) numpy array
        param voxel_size: float, voxel size for computing surface normal
        num_key_points: int, the number of key points under this map size budget
        T_init: (4,4) numpy array, the initial transformation matrix
        '''

        pcd = make_open3d_pc(pc)

        # transform back to map coordinates to compute the fpfh feature
        # since the T init is not available for offline map compression
        pcd.transform(T_init)
        radius_normal = voxel_size * 3
        print(':: Estimate normal with search radius %.3f.' % radius_normal)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,
                                                 max_nn=30))

        radius_feature = voxel_size * 5
        print(':: Compute FPFH feature with search radius %.3f.' %
              radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                                 max_nn=100))

        pcd.transform(np.linalg.inv(T_init))

        if num_key_points is not None:
            num_pts = np.asarray(pcd.points).shape[0]
            pcd = make_open3d_pc(np.asarray(pcd.points)[
                                 :min(num_pts, num_key_points), :])

            pcd_fpfh.data = pcd_fpfh.data[:, :min(num_pts, num_key_points)]

        return np.asarray(pcd.points), np.asarray(pcd_fpfh.data.T)

    def run_global_registration_ransac(self, source, target, source_feat,
                                       target_feat, num_key_points):
        '''
        param source, target: (N, 3) numpy array
        param source_feat, target_feat: (N, F) numpy array, F is depending on selected feature
        param num_key_points: int
        '''
        N0 = source.shape[0]
        N1 = target.shape[0]

        pcd0 = make_open3d_pc(source)
        pcd1 = make_open3d_pc(target)

        feat0 = make_open3d_feat(source_feat.transpose())
        feat1 = make_open3d_feat(target_feat.transpose())

        distance_threshold = 0.5

        # RANSAC 100K with 0.99 confidence
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd0, pcd1, feat0, feat1, self.cfg.mutual_filter,
            distance_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.99))

        T_est = result.transformation

        return T_est

    def run_global_registration_teaser(self, source_in, target_in, source_feat,
                                       target_feat, num_key_points):
        '''
        param source_in, target_in: (N, 3) numpy array
        param source_feat, target_feat: (N, F) numpy array, F is depending on selected feature
        param num_key_points: int
        '''
        import teaserpp_python as teaserpp

        distance_threshold = 0.5

        corres01 = match_feats(target_feat, source_feat,
                               self.cfg.mutual_filter)
        source = source_in[corres01[:, 1]]
        target = target_in[corres01[:, 0]]

        params = self.cfg.dict_methods['teaser']

        solver_params = teaserpp.RobustRegistrationSolver.Params()
        solver_params.cbar2 = params['cbar2']
        solver_params.noise_bound = params['noise_bound']
        solver_params.estimate_scaling = params['estimate_scaling']
        solver_params.rotation_tim_graph = teaserpp.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
        solver_params.rotation_estimation_algorithm = teaserpp.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        solver_params.rotation_gnc_factor = params['rotation_gnc_factor']
        solver_params.rotation_max_iterations = params[
            'rotation_max_iterations']
        solver_params.rotation_cost_threshold = params[
            'rotation_cost_threshold']

        solver = teaserpp.RobustRegistrationSolver(solver_params)
        solver.solve(source.transpose(), target.transpose())
        solution = solver.getSolution()

        T_est = np.eye(4)
        T_est[0:3, 0:3] = solution.rotation
        T_est[0:3, 3] = solution.translation

        return T_est

    def run_registration_points_icp_pt2pt_o3d(self, source, target,
                                              num_key_points):
        '''
        param source_in, target_in: (N, 3) numpy array
        param num_key_points: int
        '''
        pcd0 = make_open3d_pc(source)
        pcd1 = make_open3d_pc(target[:min(num_key_points, target.shape[0])])
        threshold = self.cfg.dict_methods['icp_pt2pt_o3d']['threshold']

        trans_init = np.eye(4)
        ransac_result = o3d.pipelines.registration.registration_icp(
            pcd0, pcd1, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        T_est = ransac_result.transformation.astype(np.float32)

        return T_est

    def run_registration_points_icp_pt2pl_o3d(self, source, target,
                                              num_key_points):
        '''
        param source_in, target_in: (N, 3) numpy array
        param num_key_points: int
        '''
        pcd0 = make_open3d_pc(source)
        pcd1 = make_open3d_pc(target)

        pcd0.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.cfg.voxel_size * 2.5, max_nn=30))

        pcd1.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.cfg.voxel_size * 2.5, max_nn=30))

        pcd1.points = pcd1.points[:min(num_key_points, target.shape[0])]
        pcd1.normals = pcd1.normals[:min(num_key_points, target.shape[0])]

        threshold = self.cfg.dict_methods[self.method]['threshold']
        trans_init = np.eye(4)

        ransac_result = o3d.pipelines.registration.registration_icp(
            pcd0, pcd1, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        T_est = ransac_result.transformation.astype(np.float32)

        return T_est
