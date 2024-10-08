import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pc_util
from torch.autograd import Function, Variable
import numpy  as np


def load_params_with_optimizer(net, filename, to_cpu=False, optimizer=None, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger.info('==> Loading parameters from checkpoint')
    checkpoint = torch.load(filename)
    epoch = checkpoint.get('epoch', -1)
    it = checkpoint.get('it', 0.0)

    net.load_state_dict(checkpoint['model_state'])

    if optimizer is not None:
        logger.info('==> Loading optimizer parameters from checkpoint')
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    logger.info('==> Done')

    return it, epoch

def load_params(net, filename, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError
    if logger is not None:
        logger.info('==> Loading parameters from checkpoint')
    checkpoint = torch.load(filename)

    net.load_state_dict(checkpoint['model_state'])
    if logger is not None:
        logger.info('==> Done')


class DBSCANCluster(Function):

    @staticmethod
    def forward(ctx, eps: float, min_pts: int, point: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param eps: float, dbscan eps
        :param min_pts: int, dbscan core point threshold
        :param point: (B, N, 3) xyz coordinates of the points
        :return:
            idx: (B, N) cluster idx
        """
        point = point.contiguous()

        B, N, _ = point.size()
        idx = torch.cuda.IntTensor(B, N).zero_() - 1

        pc_util.dbscan_wrapper(B, N, eps, min_pts, point, idx)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, grad_out):
        return ()


dbscan_cluster = DBSCANCluster.apply


class GetClusterPts(Function):

    @staticmethod
    def forward(ctx, point: torch.Tensor, cluster_idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param point: (B, N, 3) xyz coordinates of the points
        :param cluster_idx: (B, N) cluster idx
        :return:
            key_pts: (B, M, 3) cluster center pts, M is max_num_cluster_class
            num_cluster: (B, M) cluster num, num of pts in each cluster class
        """
        cluster_idx = cluster_idx.contiguous()

        B, N = cluster_idx.size()
        M = torch.max(cluster_idx) +1
        key_pts = torch.cuda.FloatTensor(B, M, 3).zero_()
        num_cluster = torch.cuda.IntTensor(B, M).zero_()
        pc_util.cluster_pts_wrapper(B, N, M, point, cluster_idx, key_pts, num_cluster)
        key_pts[key_pts * 1e4 == 0] = -1e1
        ctx.mark_non_differentiable(key_pts)
        ctx.mark_non_differentiable(num_cluster)
        return key_pts, num_cluster

    @staticmethod
    def backward(ctx, grad_out):
        return ()


get_cluster_pts = GetClusterPts.apply

def PCA(data, correlation = False, sort = True):
    '''
    data: array. The array must have N*M dimensions.
    correlation: bool.Set the type of matrix to be computed.
                 if True, compute the correlation matrix
                 if False, compute the covariance matrix
    sort: bool.Set the order that the eigenvalues/vectors will have.
        if True, they will be sorted(from higher value to less)
        if False, they won't
    :return: eigenvalues:(1, M)array.The eigenvalues of the corresponding matrix
             eigenvector:(M, M)array.The eigenvectors of the corresponding matrix
    '''
    mean = np.mean(data, axis=0)
    data_adjust = data - mean
    #the data is transposed due to np.cov/corrcoef syntax
    if correlation:
        matrix = np.corrcoef(data_adjust.T)
    else:
        matrix = np.cov(data_adjust.T)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]
        return eigenvalues, eigenvectors





def planefit(points, equation=False):
    '''

    :param points:x,y,z coordinates of the pointsset
    :param equation: bool ,if True, return the a,b,c,d foeddicients of the plane
                           if False, return 1 Points and 1 Normal vector
    a,b,c,d: float
    point, normal: array
    1 Point and 1 Normal vector: array([Px,Py,Pz]), array([Nx,Ny,Nz])
    '''
    w, v = PCA(points)
    #the normal of the plane is the last eigenvector
    normal = v[:,2]
    #get a point from the plane
    point = np.mean(points, axis=0)
    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d
    else:
        return point, normal




def point_plane_dist(points, plane):
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    FM =  np.sqrt(a**2 + b**2 + c**2)
    dist = abs(a*points[0] + b*points[1] + c*points[2] + d)/FM
    #dist = dists[0][0]
    return dist

def get_fea_centroids(embedded, masks, size):  #Tensor(N, 64) Tensor(N, I) Tensor(1)
    embedding_size = embedded.size(1)
    K = masks.size(1)
    x = embedded.unsqueeze(1).expand(-1, K, -1)  #(N, 64)->(N, I, 64)
    masks = masks.unsqueeze(2)  #(N, I, 1)
    x = x * masks
    n = size
    centroids = x[:, :n].sum(0) / masks[:, :n].sum(0)
    return centroids

def fea_distance(xyz_fea, centroid_fea):
    fea_diff = xyz_fea - centroid_fea
    distance = torch.sum(torch.abs(fea_diff), dim=-1 )
    return distance