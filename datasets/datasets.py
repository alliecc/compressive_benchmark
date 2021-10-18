
import pickle
import os
import torch


def collate_fn(list_data):
    return list_data


class FeaturePointCloudObj:
    def __init__(self, points, features=None, scores=None):
        self.points = points
        self.features = features
        self.scores = scores

    def from_tensor_to_np_array(self):

        features = self.features.numpy() if (self.features
                                             is not None) else None
        scores = self.scores.numpy() if (self.scores is not None) else None

        return FeaturePointCloudObj(self.points.numpy(), features, scores)


class LiDARDataset(torch.utils.data.Dataset):
    # Saved the local map crop and the input lidar as pairs
    def __init__(self, split, cfg):
        super().__init__()
        self.split = split
        self.cfg = cfg

        self.read_data()

    def read_data(self):
        self.load_map_data()

        path_test_samples = os.path.join(
            self.cfg.path_data,
            f"{self.split}_samples.pkl")

        path_T_gt = os.path.join(
            self.cfg.path_data,
            f"{self.split}_list_T_gt.pkl")

        print(f"Load gt pose from {path_T_gt}")
        with open(path_T_gt, 'rb') as f:
            self.list_T_gt = pickle.load(f)

        print('Loading data from %s' % path_test_samples)
        with open(path_test_samples, 'rb') as f:
            self.list_test_sample = pickle.load(f)

        self.length = len(self.list_test_sample)

        print(f"num of samples={self.length}")

    def load_map_data(self):
        self.path_preprocessed_map = os.path.join(
            self.cfg.path_data, f"{self.split}_dict_maps.pickle")

        if os.path.exists(self.path_preprocessed_map):
            print('Load map dict from: ', self.path_preprocessed_map)
            with open(self.path_preprocessed_map, 'rb') as f:
                self.dict_maps = pickle.load(f)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.list_test_sample[idx]


def get_dataloader(cfg):
    dataset = LiDARDataset('test', cfg)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             collate_fn=collate_fn)
    return dataloader
