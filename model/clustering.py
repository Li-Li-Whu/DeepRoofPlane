import torch
import  numpy as np
import torch.nn as nn
import  torch.nn.functional as F
from .model_utils import *
from utils import loss_utils
from sklearn.preprocessing import  StandardScaler
from sklearn.cluster import  DBSCAN
import queue
import  collections



def Cluster_edge(batch_dict):
        result_dir = 'result/'
        frame_id = batch_dict['frame_id']
        points = batch_dict['points']
        pts_sem = batch_dict['point_pred_sem']
        sem_label = batch_dict['sem_label']
        ins_label = batch_dict['ins_label']

        offset_pts = batch_dict['points'].clone()
        offset = batch_dict['point_pred_offset']   #(1, N, 3)
        offset_pts += 1*offset
        # offset_pts[pts_sem == 1] += offset[pts_sem == 1]

        
        pts_cluster_xyz = offset_pts.new_ones(offset_pts.shape) * -10
        pts_cluster_xyz[pts_sem == 1] = offset_pts[pts_sem == 1]  #(1, N, 3) without edge
        


        off_path = result_dir + 'offset/' + str(frame_id[0])[:-4] + '.txt'
        
        off_xyz = pts_cluster_xyz.cpu().numpy()
        off_xyz = off_xyz.squeeze()
        
        label_np = ins_label.cpu().numpy()
        label_np = label_np.squeeze()
        label_np = np.expand_dims(label_np, axis=1)
        
        off_xyz = np.concatenate((off_xyz, label_np), axis=1)
        with open(off_path, 'w') as file1:
                np.savetxt(off_path, off_xyz, fmt='%f', delimiter=' ')
        print("Write roof {} off finished!".format(str(frame_id[0])[:-4]))


        

        pts_fea = batch_dict['point_feature']
        pts_cluster_fea = pts_fea.new_ones(pts_fea.shape) * -10
        pts_cluster_fea [pts_sem == 1] = pts_fea[pts_sem == 1] # without edge
        # pts_cluster_fea = pts_fea  # with edge
        pts_cluster_fea = F.normalize(pts_cluster_fea, p=2, dim=2, eps=1e-12)
        feature_path = result_dir + 'feature/' + str(frame_id[0])[:-4] + '.txt'
        feature_np = pts_fea.cpu().numpy()
        feature_np = feature_np.squeeze()
        with open(feature_path, 'w') as file1:
                np.savetxt(feature_path, feature_np, fmt='%f', delimiter=' ')
        normfeature_path = result_dir + 'feature_norm/' + str(frame_id[0])[:-4] + '.txt'
        normfeature_np = pts_cluster_fea.cpu().numpy()
        normfeature_np = normfeature_np.squeeze()
        with open(normfeature_path, 'w+') as file1:
                np.savetxt(normfeature_path, normfeature_np, fmt='%f', delimiter=' ')
        print("Write roof {}feature finished!".format(str(frame_id[0])[:-4]))




        label_path = result_dir + 'label/' + str(frame_id[0])[:-4] + '.txt'
        coords = batch_dict['coords']
        coords_np = coords.cpu().numpy()
        coords_np = coords_np.squeeze()
        points_np = points.cpu().numpy()
        points_np = points_np.squeeze()
        
        semlabel_np = sem_label.cpu().numpy()
        semlabel_np = semlabel_np.squeeze()
        semlabel_np = np.expand_dims(semlabel_np, axis=1)
        
        inslabel_np = ins_label.cpu().numpy()
        inslabel_np = inslabel_np.squeeze()
        inslabel_np = np.expand_dims(inslabel_np, axis=1)
        label = np.concatenate((coords_np, points_np, inslabel_np, semlabel_np), axis=1)
        with open(label_path, 'w') as file1:
                np.savetxt(label_path, label, fmt='%f', delimiter=' ')
        print("Write roof {} downsample label finished!".format(str(frame_id[0])[:-4]))



        sem_path = result_dir + 'sem_pred/' + str(frame_id[0])[:-4] + '.txt'
        sem_np = pts_sem.clone().cpu().numpy()
        sem_np = sem_np.squeeze()
        sem_np = np.expand_dims(sem_np, axis=1)
        sem_np = np.concatenate((points_np, sem_np), axis=1)
        with open(sem_path, 'w') as file1:
                np.savetxt(sem_path, sem_np, fmt='%f', delimiter=' ')
        print("Write roof {} pred sem finished!".format(str(frame_id[0])[:-4]))

        



def Cluster_all(batch_dict):
        frame_id = batch_dict['frame_id']
        print(frame_id)
        points = batch_dict['points']
        pts_sem = batch_dict['point_pred_sem']

        offset_pts = batch_dict['points'].clone()
        offset = batch_dict['point_pred_offset']   #(1, N, 3)
        offset_pts += offset
        pts_cluster_xyz = offset_pts.new_ones(offset_pts.shape) * -10
        pts_cluster_xyz = offset_pts #(1, N, 3)

        pts_fea = batch_dict['point_feature']
        # pts_cluster_fea = pts_fea.new_ones(pts_fea.shape) * -10
        pts_cluster_fea = pts_fea
        pts_cluster_fea = F.normalize(pts_cluster_fea, p=2, dim=2, eps=1e-12)

        ###clustering algorithm####
        Radius = 0.40
        cluster_ptnum_thresh = 50
        sem_labels = pts_sem.squeeze()
        N = sem_labels.size(0)
        v = sem_labels.new_zeros(sem_labels.shape)
        clusters = []
        for i in range(N):
                if sem_labels[i] != 1:
                        v[i] = 1
        for i in range(N):
                if v[i] == 0:
                        Q = queue.Queue()
                        cluster_C = []
                        v[i] = 1
                        Q.put(i)
                        cluster_C.append(i)
                        if ~Q.empty():
                                k = Q.get()
                                for j in range(N):
                                        r1 = torch.dist(pts_cluster_xyz[0,j,:], pts_cluster_xyz[0,k,:])
                                        r2 = torch.dist(pts_cluster_fea[0,j,:], pts_cluster_fea[0,k,:])
                                        r = 0.1 * r1 + 0.9 * r2
                                        #if r1 < Radius:
                                        # if r2 < Radius:
                                        if r < Radius:
                                                if sem_labels[j] == sem_labels[k] and v[j] == 0:
                                                        v[j] = 1
                                                        Q.put(j)
                                                        cluster_C.append(j)
                        if len(cluster_C) > cluster_ptnum_thresh:
                                clusters.append(cluster_C)
                                #print('test1')
        print(len(clusters))
        cluster_idx = sem_labels.new_ones(sem_labels.shape) * -1
        for x in range(len(clusters)):
                for y in range(len(clusters[x])):
                        cluster_idx[clusters[x][y]] = x


        cluster_idx_np = cluster_idx.cpu().numpy()
        cluster_idx_np = cluster_idx_np.squeeze()
        cluster_idx_np = np.expand_dims(cluster_idx_np, axis=1)
        points_np = points.cpu().numpy()
        points_np = points_np.squeeze()
        
        cluster_init = np.concatenate((points_np, cluster_idx_np), axis=1)
        cluster_path = 'result/cluster_init/' + str(frame_id)[2:-6] + '.txt'
        print(cluster_path)
        #with open (cluster_path, 'w') as f:
        np.savetxt(cluster_path, cluster_init, fmt='%f', delimiter=' ')
        


        
        #dist1
        ins_num = torch.max(cluster_idx)+1
        points = points.squeeze()
        planes = []
        for i in range(ins_num):
            pointset= points[cluster_idx == i]
            pointset = pointset.cpu().numpy()
            a, b ,c, d = planefit(pointset, True)
            planes.append((a, b, c, d))

        inseg_points = points.new_zeros(points.shape)
        inseg_points[cluster_idx == -1 ] = points[ cluster_idx== -1]
        dist1 = np.ones((len(inseg_points),len(planes))) * -10
        idx = torch.where(cluster_idx == -1, cluster_idx.new_ones(cluster_idx.shape), cluster_idx.new_zeros(cluster_idx.shape))
        for m in range(len(inseg_points)):
                if idx[m] == 0:
                        continue
                for n in range(ins_num):
                        pointxyz = inseg_points[m].cpu().numpy()
                        dist1[m,n] = point_plane_dist(pointxyz, planes[n])
        #inseg_idx = np.argmax(dist,axis=1)

        #dist2
        pts_fea = batch_dict['point_feature']
        pts_fea = pts_fea.squeeze()
        #fea_centroied = pts_fea.new_zeros(pts_fea.shape)
        cluster_masks = np.zeros((points.shape[0], ins_num), dtype=np.float32)
        for f in range(len(pts_fea)):
                if cluster_idx[f] == -1:
                        continue
                cluster_masks[f, cluster_idx[f]] = 1
        cluster_masks = torch.from_numpy(cluster_masks).cuda()    #cpu.Tensor->gpu.Tensor
        centroids = get_fea_centroids(pts_fea,cluster_masks,ins_num)  #(I, 64)
        dist2 = np.ones((len(points), ins_num)) * -10
        for j in range(len(points)):
                if cluster_idx[j] != -1:
                      continue
                for k in range(ins_num):
                        pointxyz_fea = pts_fea[j]
                        centoroid_fea = centroids[k]
                        dist2[j, k] = fea_distance(pointxyz_fea, centoroid_fea)

        alpha = 1.0
        beta = 1.0
        # dist = alpha * dist1
        # dist = alpha * dist2
        dist = alpha * dist1 + beta * dist2
        ins = np.argmin(dist, axis=1)
        ins = torch.from_numpy(ins).cuda()
        for d in range(len(ins)):
                if cluster_idx[d] == -1:
                        ins[d] = ins[d]
                else:
                        ins[d] = cluster_idx[d]
        for d in range(len(ins)):
                if pts_sem[0,d] == 2:
                        ins[d] = -1
        batch_dict['point_pred_ins'] = ins

        ins_np = ins.clone().cpu().numpy()
        ins_np = ins_np.squeeze()
        ins_np = np.expand_dims(ins_np, axis=1)
        cluster_final = np.concatenate((points_np, ins_np), axis=1)
        cluster_final_path = 'result/cluster_final/' + str(frame_id)[2:-6] + '.txt'
        np.savetxt(cluster_final_path, cluster_final, fmt='%f', delimiter=' ')
        #print('test')
        


