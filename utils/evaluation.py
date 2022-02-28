#coding: utf-8
#----- 標準ライブラリ -----#
import os
from functools import lru_cache
import time
#----- 専用ライブラリ -----#
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import motmetrics as mm
import h5py
#----- 自作ライブラリ -----#
# None


class Object_Detection():
    def __init__(self, right_range=10.,
                noise_strength=0.5,
                pool_range=11):
        self.FP = 0
        self.FN = 0
        self.TP = 0

        self.right_range = right_range

        self.coordinater = self.coordinater_maximul
        # kernel_size = 4*sigma+0.5
        self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(pool_range, stride=1, padding=(pool_range - 1) // 2)
        self.noise_strength = noise_strength

    def coordinater_maximul(self, out):
        """Probability map => Position of the object. Calculate the detection position from the maximum.

        Args:
            out (tensor[batch,1,H,W]): Probability map

        Returns:
            tensor[batch,2(x,y)]: Position of the object
        """
        h = out.squeeze(1)
        h = (h >= self.noise_strength) * (h == self.maxpool(h)) * 1.0
        for _ in range(3):
            h = self.gaussian_filter(h)
            h = (h != 0) * (h == self.maxpool(h)) * 1.0

        coordinate = torch.nonzero(h)  # [detection number, 3(batch,y,x)]

        coordinate = [coordinate[coordinate[:, 0] == batch, 1:] for batch in range(h.size(0))]
        coordinate = torch.nn.utils.rnn.pad_sequence(coordinate, batch_first=True, padding_value=-1)
        #[y,x] => [x,y]
        coordinate = coordinate[:, :, [1, 0]]
        return coordinate  # [num_cell,3(batch_num,x,y)]

    def _reset(self):
        self.FP = 0
        self.FN = 0
        self.TP = 0

    def calculate(self, predict, label):
        """評価指標計算。TP,FP,FNをため込む関数。for文ないで使う

        Args:
            predict (tensor[batch,num,2(x,y)]): 予測検出位置。PADには-1が入る
            label (tensor[batch,num,2(x,y)]): 正解検出位置。PADには-1が入る
        """

        predict = predict.float()
        label = label.float()

        # もし一つも検出されなかったら
        if predict.size(0) == 0:
            for item_label in label:
                item_label = item_label[item_label[:, 0] != -1]
                self.FN += item_label.size(0)
        else:
            # 割り当て問題、どのペアを選べば距離が最も小さくなるか
            for item_pre, item_label in zip(predict, label):
                item_pre = item_pre[item_pre[:, 0] != -1]
                item_label = item_label[item_label[:, 0] != -1]

                # distanceは[batch,predict,label]となる
                distance = torch.norm(
                    item_pre[:, None, :] - item_label[None, :, :], dim=-1)

                distance = distance.to('cpu').detach().numpy().copy()

                row, col = linear_sum_assignment(distance)

                # 距離がマッチング閾値より低いかを判別
                correct_flag = distance[row, col] <= self.right_range

                # TP,FN,FPを計算
                TP = correct_flag.sum().item()

                self.TP += TP
                self.FP += item_pre.size(0) - TP
                self.FN += item_label.size(0) - TP

    def __call__(self):
        Accuracy = self.TP / (self.TP + self.FP + self.FN + 1e-7)
        Precition = self.TP / (self.TP + self.FP + 1e-7)
        Recall = self.TP / (self.TP + self.FN + 1e-7)
        F1_Score = 2 * Precition * Recall / (Precition + Recall + 1e-7)

        self._reset()
        result = {
            "Accuracy": Accuracy,
            "Precition": Precition,
            "Recall": Recall,
            "F1_Score": F1_Score
        }
        return result

    def __vars__(self):
        # 自身が定義したメソッド、変数のみ抽出
        method = {member for member in vars(
            self.__class__) if not "__" in member}

        return method

    def __dir__(self):
        # 自身が定義したメソッド、変数のみ抽出
        method = [member for member in dir(
            self.__class__) if not "__" in member]

        return method



class Object_Tracking():
    def __init__(self, right_range=10.):
        self.right_range = right_range
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, predict, label):
        """track update

        Args:
            predict (tensor[pre_track_num,4(frame,id,x,y)]): predicted track
            label (tensor[label_track_num,4(frame,id,x,y)]): label track
        """
        # frameを合わせる
        pre_min_f = predict[:, 0].min()
        pre_max_f = predict[:, 0].max()
        label_min_f = label[:, 0].min()
        label_max_f = label[:, 0].max()

        predict[:, 0] = predict[:, 0] - pre_min_f
        label[:, 0] = label[:, 0] - label_min_f
        pre_max_f = pre_max_f - pre_min_f

        for frame in range(int(pre_max_f.item())):
            pre = predict[predict[:, 0] == frame][:, [1, 2, 3]]
            lab = label[label[:, 0] == frame][:, [1, 2, 3]]

            # PAD delete
            pre = pre[pre[:, 1] >= 0].cpu().detach().float()
            lab = lab[lab[:, 1] >= 0].cpu().detach().float()
            if pre is None:
                continue

            dist_matrix = self.distance_calculate(pre, lab)

            pre_ids = pre[:, 0].long().tolist()
            lab_ids = lab[:, 0].long().tolist()

            self.acc.update(lab_ids, pre_ids, dist_matrix)

    def distance_calculate(self, predict, label):
        # predict.size() => [pre_num,3(id,x,y)],label.size() => [label_num,3(id,x,y)]

        #[label_num,pre_num]
        dist_matrix = torch.norm(label[:, None, [1, 2]] - predict[None, :, [1, 2]], dim=-1)
        dist_matrix[dist_matrix > self.right_range] = np.nan
        return dist_matrix.tolist()

    def _reset(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def __call__(self):

        calculater = mm.metrics.create()

        df = calculater.compute(self.acc,
                                metrics=mm.metrics.motchallenge_metrics,
                                name="Name")

        df_dict = df.to_dict(orient='list')

        for item in df_dict:
            df_dict[item] = df_dict[item][0]
        self._reset()
        return df_dict
