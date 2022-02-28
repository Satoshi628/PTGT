#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import sys
#----- 専用ライブラリ -----#
import hydra
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

#----- 自作ライブラリ -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.loss import connected_loss
from utils.dataset import PTC_Loader, collate_delete_PAD
import utils.transforms as tf
from utils.utils import get_movelimit
from utils.Transformer_to_track import Transformer_to_Track, Transformer_to_Track3
from utils.evaluation import Object_Tracking
from models.TATR import TATR_3D



############## dataloader関数 ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.RandomCrop(size=(256, 256)),
                                tf.RandomHorizontalFlip(p=0.5),
                                tf.RandomVerticalFlip(p=0.5)
                                ])

    val_transform = tf.Compose([])
    
    train_dataset = PTC_Loader(root_dir=cfg.PTC.root_dir, Molecule=cfg.PTC.Molecule, Density=cfg.PTC.Density, mode="train", split=cfg.PTC.split, length=cfg.PTC.length, transform=train_transform)
    val_dataset = PTC_Loader(root_dir=cfg.PTC.root_dir, Molecule=cfg.PTC.Molecule, Density="Low", mode="val", split=cfg.PTC.split, length=cfg.PTC.length, add_mode=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.val_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return train_loader, val_loader

############## train関数 ##############
def train(model, train_loader, criterion, optimizer, device):
    # モデル→学習モード
    model.train()
    #非決定論的
    torch.backends.cudnn.deterministic = False

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets, point) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        #input.size() => [batch,channel,length,H,W]
        #targets.size() => [batch,length,H,W]
        #point.size() => [batch,length,num,4(frame,id,x,y)]
        inputs = inputs.cuda(device, non_blocking=True)
        point = point.cuda(device, non_blocking=True)
        point = point.long()

        vector, coordinate = model(inputs, point[:, :, :, [2, 3]])
        
        loss = criterion(vector, point[:, :, :, 1], coordinate)

        optimizer.zero_grad()

        loss.backward()

        sum_loss += loss.item()

        del loss

        optimizer.step()
    
    torch.backends.cudnn.deterministic = True
    return sum_loss / (batch_idx + 1)

############## validation関数 ##############
def val(model, val_loader, criterion, Track_transfor, Tracking_Eva, device):
    model.eval()
    coord_F = None

    torch.backends.cudnn.deterministic = True

    with torch.no_grad():
        for batch_idx, (inputs, targets, point) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            inputs = inputs.cuda(device, non_blocking=True)
            point = point.cuda(device, non_blocking=True)
            point = point.long()

            vector, coord, coord_F = model.tracking_process(inputs, point[:, :, :, [2, 3]], add_F_dict=coord_F)

            Track_transfor.update(vector, coord)
    
    Track_transfor()
    Tracking_Eva.update(Track_transfor.get_track(), val_loader.dataset.CP_data)
    acc = Tracking_Eva()

    Track_transfor.reset()

    torch.backends.cudnn.deterministic = False
    return acc

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print(f"epoch       :{cfg.parameter.epoch}")
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    print(f"dataset     :{cfg.dataset}")
    
    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/validation.txt"

    with open(PATH_1, mode='w') as f:
        pass
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tPrecision\tRecall\tF1 Score\n")
    
    # dataset load
    train_loader, val_loader = dataload(cfg)

    move_limit = get_movelimit(train_loader.dataset, factor=1.1)

    # define model
    model = TATR_3D(back_bone_path=cfg.parameter.back_bone,
                            noise_strength=cfg.parameter.noise_strength,
                            assignment=cfg.parameter.assignment,
                            feature_num=cfg.parameter.feature_num,
                            overlap_range=cfg.parameter.overlap_range,
                            pos_mode=cfg.parameter.pos,
                            encode_mode=cfg.parameter.encoder,
                            move_limit=move_limit[0]).cuda(device)


    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    #tracking algo and accuracy
    Track_transfor = Transformer_to_Track3(tracking_mode="P", move_limit=move_limit)
    tracking_fnc = Object_Tracking()

    criterion = connected_loss(move_limit)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max)  # Adam

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
            first_cycle_steps=cfg.scheduler.first,
            cycle_mult=cfg.scheduler.mult,
            max_lr=cfg.scheduler.max,
            min_lr=cfg.scheduler.min,
            warmup_steps=cfg.scheduler.warmup,
            gamma=cfg.scheduler.gamma)
    
    torch.backends.cudnn.benchmark = True

    best_F1 = 0.0
    best_loss = 99999.
    for epoch in range(cfg.parameter.epoch):
        scheduler.step()

        # train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # validation
        if epoch >= 70:
            Tracking_acc = val(model, val_loader, criterion, Track_transfor, tracking_fnc, device)
        else:
            Tracking_acc = {'mota': 0.0, 'idp': 0.0, 'idr': 0.0, 'idf1': 0.0, 'num_switches': 0}

        ##### show result #####
        result_text = "Epoch{:3d}/{:3d} TrainLoss={:.5f} Precision={:.5f} Recall={:.5f} F1 Score={:.5f} ".format(
            epoch + 1,
            cfg.parameter.epoch,
            train_loss,
            Tracking_acc['idp'], Tracking_acc['idr'], Tracking_acc['idf1'])

        print(result_text)

        ##### write result #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
        with open(PATH_2, mode='a') as f:
            f.write("{0}\t{1[mota]:.4f}\t{1[idp]:.4f}\t{1[idr]:.4f}\t{1[idf1]:.4f}\t{1[num_switches]}\n".format(epoch + 1, Tracking_acc))
        
        if Tracking_acc['idf1'] > best_F1:
            best_F1 = Tracking_acc['idf1']
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)

    print("Best F1Score:{:.4f}%".format(best_F1))

############## main ##############
if __name__ == '__main__':
    main()
    
