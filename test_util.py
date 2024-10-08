import glob
import tqdm
import os
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools
from model.model_utils import *
from os.path import join,exists,dirname,abspath

BASE_DIR = dirname(abspath(__file__))

def writePoints(points, sem_pred, off_pts, fea, clsRoad):
    points = np.squeeze(points)
    offset = np.squeeze(off_pts)
    fea = np.squeeze(fea)
    off_pts = points + offset
    sem_pred = np.transpose(sem_pred)
    point_off = np.concatenate((points, off_pts), axis = 1)
    point_off_sem = np.concatenate((point_off, sem_pred), axis=1)
    point_off_sem_fea =  np.concatenate((point_off_sem, fea), axis = 1)
    point_off_sem_fea = point_off_sem_fea.tolist()
    with open(clsRoad, 'w+') as file1:
        np.savetxt(clsRoad, point_off_sem_fea, fmt = '%f', delimiter=' ')
            #file1.writelines("\n".join([" ".join([i for i in j]) for j in point_sem]))

def writepoints(points, sem_label, ins_label, clsRoad):
    points = np.squeeze(points)
    #offset = np.squeeze(offset)
    #sem_label = np.transpose(sem_label)
    sem_label = np.squeeze(sem_label)
    #off_pts = points + offset
    sem_label = np.expand_dims(sem_label, axis=1)
    ins_label = np.squeeze(ins_label)
    ins_label = np.expand_dims(ins_label, axis=1)
    points_ins = np.concatenate((points, ins_label), axis = 1)
    points_ins_sem = np.concatenate((points_ins, sem_label), axis=1)
    points_ins_sem = points_ins_sem.tolist()
    with open(clsRoad, 'w+') as file1:
        #np.savetxt(clsRoad, points, fmt = '%f', delimiter=' ')
        np.savetxt(clsRoad, points_ins_sem, fmt='%f', delimiter=' ')

def writeoffset(points, sem_pred, offset, pred_ins, clsRoad):
    points = np.squeeze(points)
    #offset = np.squeeze(offset)
    #sem_pred = np.transpose(sem_pred)
    #off_pts = points + offset
    #pred_ins = np.expand_dims(pred_ins, axis=0)
    #pred_ins = np.transpose(pred_ins)
    #points_off = np.concatenate((points, off_pts), axis = 1)
    #points_off_sem = np.concatenate((points_off, sem_pred), axis=1)
    #points_off_sem_ins = np.concatenate((points_off_sem, pred_ins), axis=1)
    #points_off_sem_ins = points_off_sem_ins.tolist()
    points = points.tolist()
    with open(clsRoad, 'w+') as file1:
        #np.savetxt(clsRoad, points_off_sem_ins, fmt = '%f', delimiter=' ')
        np.savetxt(clsRoad, points, fmt='%f', delimiter=' ')

def writefeature(feature, clsRoad):
    feature = np.squeeze(feature)
    feature = feature.tolist()
    with open(clsRoad, 'w+') as file1:
        np.savetxt(clsRoad, feature, fmt='%f', delimiter=' ')

def writesemins(points, sem_pred,  pred_ins, clsRoad):
    points = np.squeeze(points)
    #offset = np.squeeze(offset)
    sem_pred = np.transpose(sem_pred)
    #off_pts = points + offset
    pred_ins = np.expand_dims(pred_ins, axis=0)
    pred_ins = np.transpose(pred_ins)
    #points_off = np.concatenate((points, off_pts), axis = 1)
    points_sem = np.concatenate((points, sem_pred), axis= 1)
    #points_off_sem = np.concatenate((points_off, sem_pred), axis=1)
    points_sem_ins = np.concatenate((points_sem, pred_ins), axis=1)
    points_sem_ins = points_sem_ins.tolist()
    with open(clsRoad, 'w+') as file1:
        np.savetxt(clsRoad, points_sem_ins, fmt = '%f', delimiter=' ')


def test_model(model, data_loader, logger, test_tag):
    #bias_path = BASE_DIR + '/output/' + test_tag + '/bias.txt'

    with tqdm.trange(0, len(data_loader), desc='test', dynamic_ncols=True) as tbar:
        for cur_it in tbar:
            batch = next(dataloader_iter)
            load_data_to_gpu(batch)
            with torch.no_grad():
                batch = model(batch)
                load_data_to_cpu(batch)
        
        print("test finished!")

        

