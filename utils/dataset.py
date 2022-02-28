#coding: utf-8
#----- 標準ライブラリ -----#
import re
import os
import random
import glob
#----- 専用ライブラリ -----#
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
#----- 自作モジュール -----#
#None

PAD_ID = -1


##### PTC dataset #####
class PTC_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/PTC", Molecule="RECEPTOR", Density="Low", mode="train", split=0, length=16,add_mode=False, transform=None):
        
        split_list = [{"train": [0, 1, 2], "val": [3], "test": [4]},
                      {"train": [4, 0, 1], "val": [2], "test": [3]},
                      {"train": [3, 4, 0], "val": [1], "test": [2]},
                      {"train": [2, 3, 4], "val": [0], "test": [1]},
                      {"train": [1, 2, 3], "val": [4], "test": [0]}]
        
        #エラー対策
        if not Density in use_data_list:  # キーがあるかどうか
            raise KeyError("Densityが{0},{1},{2}以外の文字になっています。".format(*list(use_data_list.keys())))
        if not mode in split_list[0]:  # キーがあるかどうか
            raise KeyError("modeが{0},{1},{2}以外の文字になっています。".format(*list(split_list[0].keys())))

        self.length = length

        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, Molecule, use_data_list[Density][idx]) for idx in split_list[split][mode]]
        use_frames = 0

        
        img_paths = []
        EP_paths = []
        for path in use_paths:
            #相対パスを追加
            img_paths.append(sorted(glob.glob(path + "/*.tif")))
            #imgパスからmpmパスを取得する
            EP_paths.append(re.search(r'Sample\d+', path).group())
        
        #読み込み時すべてnumpyより、list内部numpyの方が処理が早い
        #image保存、正規化も行う(0~1の値にする)、大きさ[1,H,W]となる。
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]

        #EP保存
        EP_images = [h5py.File("{}/{}/Molecule_EP.hdf5".format(root_dir, Molecule), mode='r')[path] for path in EP_paths]
        EP_images = [list(EP.values()) for EP in EP_images]
        
        #削除するindexを求めるlength-1コ消す
        delete_index = [len(EP) for EP in EP_images]
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]

        #EP_imageをすべてメモリに保存する。
        self.EP_images = [image[...] for EP in EP_images for image in EP]
        
        #Cell Point保存
        CP_data = [h5py.File("{}/{}/Cell_Point_Annotation.hdf5".format(root_dir, Molecule), mode='r')[path][...] for path in EP_paths]

        self.CP_data_per_data = [torch.tensor(data) for data in CP_data]
        #結合
        self.CP_data = [sample_data[sample_data[:, 0] == frame] for sample_data in self.CP_data_per_data for frame in torch.unique(sample_data[:, 0])]
        self.CP_batch_data = torch.nn.utils.rnn.pad_sequence(self.CP_data, batch_first=True, padding_value=PAD_ID)
        self.CP_data = torch.cat(self.CP_data,dim=0)


        #getitemのindexとEP_igamesのindexを合わせる配列の作製
        self.idx_transfer = list(range(len(self.EP_images)))
        #最後のidxから削除
        for idx in delete_index[::-1]:
            self.idx_transfer.pop(idx)

        #val,testは同じ画像を入力する必要がない追跡アルゴリズムの場合
        #1iter 0,1,2,3, 2iter 4,5,6,7,...とする
        if mode == "val" or mode == "test":
            length_overlap = add_mode * 1
            self.idx_transfer = [idx for idx in self.idx_transfer if idx % (length - length_overlap) == 0]

        #transforme
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        
        #imageは[length,H,W]、labelは[length,H,W]、pointは[length,mol_max_number,4(frame,id,x,y)]
        image = np.concatenate([self.images[data_index + length] for length in range(self.length)], axis=0)
        label = np.concatenate([self.EP_images[data_index + length] for length in range(self.length)], axis=0)
        point = torch.stack([self.CP_batch_data[data_index + length] for length in range(self.length)], dim=0)

        # もしself.transformが有効なら
        if self.transform:
            image, label, point = self.transform(image, label, point)

        return image[None], label, point  #image[1,length,H,W],label[length,H,W],point[length,num,4(frame,id,x,y)]

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_track(self):
        return self.CP_data_per_data


def collate_delete_PAD(batch):
    image, label, point = list(zip(*batch))

    image = torch.stack(image)
    label = torch.stack(label)
    point = torch.stack(point)

    #PADをなるべく少なくする
    sorted_idx = torch.argsort(point[:, :, :, 0], dim=2, descending=True)
    point = torch.take_along_dim(point, sorted_idx[:, :, :, None], dim=2)
    max_point_num = (point[:, :, :, 0] != -1).sum(dim=-1).max()
    point = point[:, :, :max_point_num]

    return image, label, point
