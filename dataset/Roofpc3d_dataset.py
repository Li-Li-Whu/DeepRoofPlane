import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

def read_pts(pts_file):
    with open(pts_file, 'r') as f:
        lines = f.readlines()
        pts = np.array([f.strip().split(' ') for f in lines], dtype=np.float64)
    return pts

class Roofpc3dDataset(Dataset):
    def __init__(self, data_path, transform, npoint, logger=None, noise=False):
        #print(data_path)
        with open(data_path, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [f.strip() for f in self.file_list]
            flist = []
            for l in self.file_list:
                flist.append(l)
            self.file_list = flist

            self.npoint = npoint #npoint  #NPOINT=3000

            self.transform = transform

            self.noise = noise

            if logger is not None:
                logger.info('TOtal samples: %d' % len(self))


    def __len__(self):
        return  len(self.file_list)


    def __getitem__(self, item):
        file_path = self.file_list[item]
        fram_id = file_path.split('/')[-1]
        print(fram_id)
        xyzlabel = list(torch.load(file_path))
        #print(xyzlabel)
       
        points = xyzlabel[0]
        ins_label = xyzlabel[1]
        sem_label = xyzlabel[2]
        coords = points
        #print(sem_label)
        #assert 1==0, "error"
        '''
        sem_label = np.ones(ins_label.shape, dtype = np.int) * -10
        for i in range(len(sem_label)):
            if ins_label[i] == -1 :
                sem_label[i] = 0
            else:
                sem_label[i] = 1
        '''

        points = self.transform(points)
        

        min_pt, max_pt = np.min(points, axis =0), np.max(points, axis = 0)
        maxXYZ = np.max(max_pt)
        minXYZ = np.min(min_pt)
        min_pt[:] = minXYZ
        max_pt[:] = maxXYZ
        points = (points - min_pt) / (max_pt - min_pt)
        '''
        with open("result/trans/" + fram_id +'_norm.txt', 'w') as f:
            for id, itm in enumerate(points):
                line = str(itm[0]) + ' ' + str(itm[1]) + ' ' + str(itm[2]) + ' ' + str(ins_label[id]) + '\n'
                f.write(line)
        f.close()
        '''

        offsets = np.ones((points.shape[0], 3), dtype=np.float32)
        inst_num = int(ins_label.max()) + 1
        pl_centers = np.ones((inst_num, 4), dtype=np.float32)
        for l in range (inst_num):
            idx_ins_i = np.where(ins_label == l)
            xyz_ins_i = points[idx_ins_i]
            centroid = xyz_ins_i.mean(0)
            offsets[idx_ins_i, :] = centroid - xyz_ins_i
            pl_centers[l, :3] = centroid
            pl_centers[l, 3] = l


        if len(points) > self.npoint:
            idx = np.random.randint(0, len(points), self.npoint)
        else:
            idx = np.random.randint(0, len(points), self.npoint - len(points))
            idx = np.append(np.arange(0, len(points)), idx)
        np.random.shuffle(idx)

        points = points[idx]
        offsets = offsets[idx]
        ins_label = ins_label[idx]
        sem_label = sem_label[idx]
        #inst_info = inst_info[idx]
        coords = coords[idx]

        ins_label = np.unique(ins_label, False, True)[1]
        #max_instances = np.amax(ins_label) + 1
        max_instances = 10
        masks = np.zeros((points.shape[0], max_instances), dtype = np.float32)
        masks[np.arange(points.shape[0]), ins_label[:]] = 1

        points = points.astype(np.float32)
        coords = coords.astype(np.float32)
        offsets = offsets.astype(np.float32)
        min_pt = min_pt.astype(np.float32)
        max_pt = max_pt.astype(np.float32)
        ins_label = ins_label.astype(np.int32)
        sem_label = sem_label.astype(np.int32)
        pt = np.concatenate(( np.expand_dims(min_pt, 0), np.expand_dims(max_pt, 0)), axis = 0 )
        data_dict = {'points': points, 'coords': coords, 'offset': offsets, 'ins_label': ins_label, 'sem_label': sem_label, 'masks': masks,
                     'size': np.unique(ins_label[:]).size, 'frame_id': fram_id, 'minMaxPt': pt}
        return data_dict

    def getInstanceInfo(self, xyz, instance_label, frame_id):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        r = np.ones((xyz.shape[0],1), dtype=np.float32) * -100.0     #(n,1)
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            r_ =  np.linalg.norm((max_xyz_i - min_xyz_i), ord=2)
            r_ = r_ / 2
            r[inst_idx_i] = r_
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i
        return instance_num, {"instance_info": instance_info, 'r': r}

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret= {}
        for key, val in data_dict.items():
            try:
                if key == 'points':
                    ret[key] = np.concatenate(val, axis = 0).reshape([batch_size, -1, val[0].shape[-1]])
                elif key == 'coords':
                    ret[key] = np.concatenate(val, axis=0).reshape([batch_size, -1, val[0].shape[-1]])
                elif key == 'offset':
                    ret[key] = np.concatenate(val, axis=0).reshape([batch_size, -1, val[0].shape[-1]])
                elif key in ['ins_label', 'sem_label']:
                    ret[key] = np.concatenate(val, axis = 0).reshape([batch_size, -1, val[0].shape[-1]])
                elif key in ['masks']:
                    ret[key] = np.concatenate(val, axis = 0).reshape([batch_size, -1, val[0].shape[-1]])
                elif key in ['size']:
                    ret[key] = val
                elif key in ['frame_id']:
                    ret[key] = val
                elif key in ['minMaxPt']:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret














