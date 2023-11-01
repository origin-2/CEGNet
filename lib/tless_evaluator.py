#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import torch
import torch.nn.parallel
import concurrent.futures
import numpy as np
import torch.utils.data


cls_num = 31
cls_lst = [i for i in range(cls_num)] # cls[0] means all classes
cls_name = ['1', '2', '3', '4', '5',
            '6', '7', '8', '9', '10',
            '11', '12', '13', '14', '15', '16',
            '17', '18', '19', '20', '21',
            '22', '23', '24', '25', '26',
            '27', '28', '29', '30']
log_eval_dir = './'

class TLESSADDval():

    def __init__(self):

        n_cls = cls_num
        self.n_cls = cls_num
        self.cls_add_dis = [list() for i in range(n_cls)]  
        self.cls_adds_dis = [list() for i in range(n_cls)] 
        self.cls_add_s_dis = [list() for i in range(n_cls)] 
        sym_cls_ids = [0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   13,   14,   15,   16,   18,   19,   22,   23,   24,   25,   26,   27,   28,   29]
        self.sym_cls_ids = [i+1 for i in sym_cls_ids]
        self.log_eval_dir = log_eval_dir


    def cal_auc(self):

        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []

        add_2cm_lst = []
        adds_2cm_lst = []
        add_s_2cm_lst = []

        for cls_id in range(1, self.n_cls): 


            if (cls_id) in self.sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]


        for i in range(self.n_cls):

            add_auc, add_2cm = cal_auc(self.cls_add_dis[i], max_dis=0.05, thr_m=0.01) 
            adds_auc, adds_2cm = cal_auc(self.cls_adds_dis[i], max_dis=0.05, thr_m=0.01) 
            add_s_auc, add_s_2cm = cal_auc(self.cls_add_s_dis[i], max_dis=0.05, thr_m=0.01) 

            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)

            add_2cm_lst.append(add_2cm)
            adds_2cm_lst.append(adds_2cm)
            add_s_2cm_lst.append(add_s_2cm)
            if i == 0:
                continue
            print(cls_name[i-1])
            print("**** add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(add_auc, adds_auc, add_s_auc))
            print("<2cm add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(add_2cm, adds_2cm, add_s_2cm))

        print("Average of all object:")
        print("**** add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(np.mean(add_auc_lst[1:]), np.mean(adds_auc_lst[1:]), np.mean(add_s_auc_lst[1:])))
        print("<2cm add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(np.mean(add_2cm_lst[1:]), np.mean(adds_2cm_lst[1:]), np.mean(add_s_2cm_lst[1:])))

        print("All object :")
        print("**** add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(add_auc_lst[0], adds_auc_lst[0], add_s_auc_lst[0]))
        print("<2cm add: {:.2f}, adds: {:.2f}, add(-s): {:.2f}".format(add_2cm_lst[0], adds_2cm_lst[0], add_s_2cm_lst[0]))


        return {'auc': add_s_auc_lst[0]}

    def eval_pose_parallel(self, pred_RT_lst, pred_clsID_lst, gt_RT_lst, gt_clsID_lst, models_pts_lst):

        bs = len(pred_clsID_lst)

        with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as executor:

            for res in executor.map(eval_metric, pred_RT_lst, pred_clsID_lst, gt_RT_lst, gt_clsID_lst, models_pts_lst):

                cls_add_dis_lst, cls_adds_dis_lst = res
                self.cls_add_dis = self.merge_lst( self.cls_add_dis, cls_add_dis_lst)
                self.cls_adds_dis = self.merge_lst(self.cls_adds_dis, cls_adds_dis_lst)


    def merge_lst(self, targ, src):

        for i in range(len(targ)):
            targ[i] += src[i]

        return targ


def eval_metric(pred_RT, pred_cls_id, gt_RT, gt_cls_id, models_pts):  

    n_cls = cls_num
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    if pred_cls_id[0] == 0 or pred_cls_id[0] != gt_cls_id[0]:
        return cls_add_dis, cls_adds_dis

    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = torch.from_numpy(gt_RT.astype(np.float32)).cuda()
    mesh_pts = torch.from_numpy(models_pts.astype(np.float32)).cuda()
    add = cal_add_cuda(pred_RT, gt_RT, mesh_pts) 
    adds = cal_adds_cuda(pred_RT, gt_RT, mesh_pts) 
    cls_add_dis[pred_cls_id[0]].append(add.item())
    cls_adds_dis[pred_cls_id[0]].append(adds.item())
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())



    return cls_add_dis, cls_adds_dis



def cal_add_cuda(pred_RT, gt_RT, p3ds):  
    _, N = p3ds.size()
    pred_p3ds = torch.mm(pred_RT[:, :3], p3ds) + pred_RT[:, 3].view(3, 1).repeat(1, N)
    gt_p3ds = torch.mm(gt_RT[:, :3], p3ds) + gt_RT[:, 3].view(3, 1).repeat(1, N)
    dis = torch.norm(pred_p3ds - gt_p3ds, dim=0)
    return torch.mean(dis)


def cal_adds_cuda(pred_RT, gt_RT, p3ds):  

    _, N = p3ds.size()
    pd = torch.mm(pred_RT[:, :3], p3ds) + pred_RT[:, 3].view(3, 1).repeat(1, N)
    pd = pd.view(1, 3, N).repeat(N, 1, 1).permute(2, 1, 0)
    gt = torch.mm(gt_RT[:, :3], p3ds) + gt_RT[:, 3].view(3, 1).repeat(1, N)
    gt = gt.view(1, 3, N).repeat(N, 1, 1)
    dis = torch.norm(pd - gt, dim=1)
    mdis = torch.min(dis, dim=1)[0]

    return torch.mean(mdis)


def cal_auc(add_dis, max_dis=0.1, thr_m = 0.02):

    D = np.array(add_dis)
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(add_dis)
    acc = np.cumsum(np.ones((1, n)), dtype=np.float32) / n 
    aps = VOCap(D, acc) 

    add_t_cm = np.where(D < thr_m)[0].size / D.size

    return aps * 100, add_t_cm * 100



def VOCap(rec, prec):
    idx = np.where(rec != np.inf)  
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx] 
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap
