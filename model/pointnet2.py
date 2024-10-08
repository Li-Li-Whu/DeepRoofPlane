import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import loss_utils
from model.model_utils import *
#from torchstat import stat
import time


def square_distance(src, dst):
    """
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # 2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1) # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) # xm*xm + ym*ym + zm*zm
    return dist



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):

        centroids[:, i] = farthest

        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)

        dist = torch.sum((xyz - centroid) ** 2, -1)
        
        mask = dist < distance
        distance[mask] = dist[mask]
        
        farthest = torch.max(distance, -1)[1]
    return centroids



def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1,...DN]
    Return:
        new_points:, indexed points data, [B, D1,...DN, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    sqrdists = square_distance(new_xyz, xyz)

    group_idx[sqrdists > radius ** 2] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])

    mask = group_idx == N
  
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint: Number of point for FPS
        radius: Radius of ball query
        nsample: Number of point for each ball query
        xyz: Old feature of points position data, [B, N, C]
        points: New feature of points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape

    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))

    idx = query_ball_point(radius, nsample, xyz, new_xyz)

    grouped_xyz = index_points(xyz, idx)

    grouped_xyz -= new_xyz.view(B, npoint, 1, C)
 
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points  = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        ''' 
        PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius_list: list of float32 -- search radius in local region
            nsample_list: list of int32 -- how many points in each local region
            mlp_list: list of list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, sum_k{mlp[k][-1]}) TF tensor
    '''
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  #(B, S, C)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  #(B, S, K)
            grouped_xyz = index_points(xyz, group_idx)  # (B, N, C) -> (B, S, K, C )
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)  #(B, N, D) -> (B, S, K ,D)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1) #
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat



class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        return new_points

class PointNet2(nn.Module):
    def __init__(self, model_cfg, in_channel=3):
        super().__init__()
        self.model_cfg = model_cfg
        self.noise = self.model_cfg.NOISE
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], in_channel,[[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256],[128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 256, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512 + 256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)

        self.conv_o1 = nn.Conv1d(128, 64, 1)
        self.bn_o1 = nn.BatchNorm1d(64)
        self.conv_o2 = nn.Conv1d(64, 32, 1)
        self.bn_o2 = nn.BatchNorm1d(32)
        self.offset_branch = nn.Conv1d(32, 3, 1)
        #RoofN3d with noise(sem_label:0,1,2), Roofpc3d without noise(sem_label:0,1)
        if self.noise:
            self.conv_s1 = nn.Conv1d(128, 64, 1)
            self.bn_s1 = nn.BatchNorm1d(64)
            self.conv_s2 = nn.Conv1d(64, 32, 1)
            self.bn_s2 = nn.BatchNorm1d(32)
            self.sem_branch = nn.Conv1d(32, 3, 1)
        else:
            self.conv_s1 = nn.Conv1d(128, 64, 1)
            self.bn_s1 = nn.BatchNorm1d(64)
            self.conv_s2 = nn.Conv1d(64, 32, 1)
            self.bn_s2 = nn.BatchNorm1d(32)
            self.sem_branch = nn.Conv1d(32, 2, 1)
        self.conv_f1 = nn.Conv1d(128, 64, 1)
        self.bn_f1 = nn.BatchNorm1d(64)
        self.conv_f2 = nn.Conv1d(64, 64, 1)
        self.bn_f2 = nn.BatchNorm1d(64)
        self.fea_branch = nn.Conv1d(64, 64, 1)
        self.init_weights()
        self.num_output_features = 128
        if self.training:
            self.train_dict = {}
            self.add_module(
                'sem_loss_func',
                torch.nn.CrossEntropyLoss()
            )
            self.add_module(
                'insfea_loss_func',
                loss_utils.DiscriminativeLoss( delta_d=1.5, delta_v=0.5)
            )
            self.loss_weight = self.model_cfg.LossWeight

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        xyz = batch_dict['points']
        if self.training :
            sem_label = batch_dict['sem_label']
            ins_label = batch_dict['ins_label']
            coords = batch_dict['coords']
            #instance_info = batch_dict['inst_info']
            gt_offset = batch_dict['offset']
            masks = batch_dict['masks']
            size = batch_dict['size']
            self.train_dict.update({
                'offset_label': gt_offset,  #(B, N, C)
                'sem_label': sem_label,  #(B, N)
                'ins_label': ins_label,  #(B, N)
                'masks': masks,   # (B, N, I)
                'size': size
            })
        # start_time = time.time()
        fea = xyz
        print(xyz.size())
        l0_fea = fea.permute(0, 2, 1)
        l0_xyz = l0_fea

        l1_xyz, l1_fea = self.sa1(l0_xyz, l0_fea)
        l2_xyz, l2_fea = self.sa2(l1_xyz, l1_fea)
        l3_xyz, l3_fea = self.sa3(l2_xyz, l2_fea)
        l4_xyz, l4_fea = self.sa4(l3_xyz, l3_fea)

        l3_fea = self.fp4(l3_xyz, l4_xyz, l3_fea, l4_fea)
        l2_fea = self.fp3(l2_xyz, l3_xyz, l2_fea, l3_fea)
        l1_fea = self.fp2(l1_xyz, l2_xyz, l1_fea, l2_fea)
        l0_fea = self.fp1(l0_xyz, l1_xyz, None, l1_fea)

        x = self.drop1(self.bn1(self.conv1(l0_fea)))
        o = F.relu(self.bn_o2(self.conv_o2(F.relu(self.bn_o1(self.conv_o1(x))))))
        s = F.relu(self.bn_s2(self.conv_s2(F.relu(self.bn_s1(self.conv_s1(x))))))
        f = F.relu(self.bn_f2(self.conv_f2(F.relu(self.bn_f1(self.conv_f1(x))))))
        pred_offset = self.offset_branch(o).permute(0, 2, 1)  #(B, N, C)
        pred_sem = self.sem_branch(s).permute(0, 2, 1)  #(B, N, 1)
        pred_fea = self.fea_branch(f).permute(0, 2, 1) #(B, N, 64)
        softmax = torch.nn.Softmax(dim=-1)
        pred_sem_scores = softmax(pred_sem).squeeze(-1)
        pred_sem_values, pred_sem_indices = torch.max(pred_sem_scores, dim=-1, keepdim=False)
        # end_time = time.time()
        # duration = end_time - start_time
        # print("forward propagation cost time：", duration, "s")
        if self.training:
            self.train_dict.update({
                'sem_pred': pred_sem_indices,
                'sem_pred_scores': pred_sem_scores,
                'offset_pred': pred_offset,
                'pts_fea': pred_fea  #(B, N, 64)
            })
        batch_dict['point_feature'] = pred_fea   #(B, N, 64)
        batch_dict['point_pred_score'] = pred_sem_scores  #(B, N)
        batch_dict['point_pred_sem'] = pred_sem_indices #(B, N)
        batch_dict['point_pred_offset'] = pred_offset  #(B, N, C)

        return batch_dict

    def loss(self, loss_dict, disp_dict):
        # pred_sem = self.train_dict['sem_pred_scores']
        pred_sem, pred_offset = self.train_dict['sem_pred_scores'], self.train_dict['offset_pred']
        # label_sem = self.train_dict['sem_label']
        label_sem, label_offset = self.train_dict['sem_label'], self.train_dict['offset_label']
        sem_loss = self.get_sem_loss(pred_sem, label_sem, self.loss_weight['sem_weight'])
        off_norm_loss, off_dir_loss = self.get_off_loss(pred_offset, label_offset, self.loss_weight['off_norm_weight'], self.loss_weight['off_dir_weight'])
        
        embedded = self.train_dict['pts_fea']
        masks, size = self.train_dict['masks'], self.train_dict['size']
        ptsfea_loss = self.insfea_loss_func(embedded, masks, size)


        loss = sem_loss + off_norm_loss + off_dir_loss + ptsfea_loss
        loss_dict.update({
            'pts_sem_loss': sem_loss.item(),
            'pts_off_norm_loss': off_norm_loss.item(),
            'pts_off_dir_loss': off_dir_loss.item(),
            'pts_fea_loss': ptsfea_loss.item(),
            'loss': loss.item()
        })


        pred_sem_i = self.train_dict['sem_pred']
        pred_sem_i = pred_sem_i.squeeze(-1)  #(B,N)
        label_sem = label_sem.permute(0, 2, 1).squeeze(-1)  #(B,N)
        OA = torch.sum((pred_sem_i == label_sem)).item() / len(label_sem.view(-1))

        disp_dict.update({'pts_OA': OA
                          #'pts_CA0':CA0,
                          #'pts_CA1':CA1,
                          #'pts_CA2':CA2,
                          #'pts_mAcc':mAcc
                        })
        return loss, loss_dict, disp_dict




    def get_sem_loss(self, pred, label, weight):
        batch_size = int(pred.shape[0])
        numN = int(pred.shape[1])
        sem_loss_src = 0
        for a in range(batch_size):
            pred_ = pred[a,:,:]
            label_= label[a,:,:]
            label_= label_.permute(1, 0).squeeze(-1) #(N, 3)
            sem_loss_src_ = self.sem_loss_func(pred_, label_.long())
            sem_loss_src += sem_loss_src_
        sem_loss = sem_loss_src/batch_size
        sem_loss = sem_loss * weight
        return sem_loss

    
    def get_off_loss(self, pred, label, weight1, weight2):
        batch_size = int(pred.shape[0])
        numN = int(pred.shape[1])
        pt_diff = pred - label     # (B, N, 3 )
        pt_dist = torch.sum(torch.abs(pt_diff), dim =-1)  #(B, N)
        offset_norm_loss = torch.sum(pt_dist) / ( batch_size * numN )
        offset_norm_loss = offset_norm_loss * weight1

        gt_offset_norm = torch.norm(label, p=2, dim=2) #(N), float
        gt_offset_ = label / (gt_offset_norm.unsqueeze(-1) + 1e-8)
        pt_ofset_norm = torch.norm(pred, p=2, dim=2)
        pt_offsets_ = pred / (pt_ofset_norm.unsqueeze(-1) + 1e-8)
        direction_diff = -(gt_offset_ * pt_offsets_).sum(-1)
        offset_dif_loss = torch.sum(direction_diff) / ( batch_size * numN )  #(B, N)
        offset_dif_loss = offset_dif_loss * weight2

        return offset_norm_loss, offset_dif_loss














