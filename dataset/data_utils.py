import numpy as np
import os
import sys

from torch.utils.data import DataLoader
from dataset.Roofpc3d_dataset import Roofpc3dDataset
from dataset.RoofN3d_dataset import RoofN3dDataset


__all__ = {
    'Roofpc3dDataset': Roofpc3dDataset,
    'RoofN3dDataset': RoofN3dDataset
}

class GaussianTransform:
    def __init__(self, sigma = (0.005, 0.015), clip = 0.05, p = 0.8):
        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, points):
        if np.random.rand(1) < self.p:
            lastsigma = np.random.rand(1) * (self.sigma[1] - self.sigma[0]) + self.sigma[0]
            row, Col = points.shape
            jittered_point = np.clip(lastsigma * np.random.randn(row, Col), -1 * self.clip, self.clip)
            jittered_point += points
            return jittered_point
        else:
            return  points

def build_dataloader(path, dataset, batch_size,  data_cfg, noise=False, workers=16, logger=None, training=True):
    if training:
        dataset_path = path + '/' + dataset + '/train.txt'
    else:
        dataset_path = path + '/' + dataset + '/test.txt'
    #print(path)

    npoint = data_cfg.NPOINT
    noise = data_cfg.NOISE
    if dataset == 'Roofpc3d':
        if training:
            trasform = GaussianTransform(sigma=(0.010, 0.080), clip=10, p=0.9)
        else:
            trasform = GaussianTransform(sigma=(0.010, 0.050), clip=10, p=0.0)
        dataset = Roofpc3dDataset(dataset_path, trasform, npoint, logger, noise)
    elif  dataset == 'RoofN3d':
        dataset = RoofN3dDataset(dataset_path, npoint, logger, noise)
    else:
        print("Error: no data loader")


    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers, collate_fn=dataset.collate_batch,
        shuffle=training)
    return dataloader

