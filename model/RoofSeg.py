from .pointnet2 import PointNet2
import  torch.nn as nn
from .clustering import Cluster_edge
from .clustering import Cluster_all


class RoofSeg(nn.Module):
    def __init__(self, model_cfg, input_channel =3, cluster=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = PointNet2(model_cfg.PointNet2, input_channel)
        self.cluster = cluster #False
        #self.cluster_net = Cluster(model_cfg.ClusterNet)

    def forward(self, batch_dict):
        batch_dict = self.backbone(batch_dict)
        if self.training:
            loss = 0
            loss_dict = {}
            disp_dict = {}
            tmp_loss, loss_dict, disp_dict = self.backbone.loss(loss_dict, disp_dict)
            loss += tmp_loss
            return loss, loss_dict, disp_dict
        else:
            if not self.cluster:
                Cluster_edge(batch_dict)
            else:
                Cluster_all(batch_dict)
            return batch_dict

