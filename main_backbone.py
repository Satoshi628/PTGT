#coding: utf-8
#----- 標準ライブラリ -----#
import hydra
import os
import argparse
import random

#----- 専用ライブラリ -----#
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
from utils.dataset import PTC_Loader, collate_delete_PAD
import utils.transforms as tf
from utils.utils import get_movelimit
from utils.evaluation import Object_Detection
from models.Unet_3D import UNet_3D, UNet_3D_FA
from utils.loss import Contrastive_Loss

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.RandomCrop(size=(256, 256)),
                                tf.RandomHorizontalFlip(p=0.5),
                                tf.RandomVerticalFlip(p=0.5),
                                ])

    val_transform = tf.Compose([])
    
    train_dataset = PTC_Loader(root_dir=cfg.PTC.root_dir, Molecule=cfg.PTC.Molecule, Density="Mid", mode="train", split=cfg.PTC.split, length=cfg.PTC.length, transform=train_transform)
    val_dataset = PTC_Loader(root_dir=cfg.PTC.root_dir, Molecule=cfg.PTC.Molecule, Density="Low", mode="val", split=cfg.PTC.split, length=cfg.PTC.length, transform=val_transform)

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
    model.train()

    sum_loss = 0

    for batch_idx, (inputs, targets, point) in enumerate(tqdm(train_loader, leave=False)):
        #input.size() => [batch,channel,length,H,W]
        #targets.size() => [batch,length,H,W]
        #point.size() => [batch,length,num,4(frame,id,x,y)]

        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)
        point = point.cuda(device, non_blocking=True)
        point = point.long()


        predicts, feature, coord = model(inputs, point[:, :, :, 2:])
        loss = criterion(predicts, feature, coord, targets, point[:, :, :, 1])

        optimizer.zero_grad()

        # back propagation
        loss.backward()

        sum_loss += loss.item()

        del loss  # Memory saving

        optimizer.step()
    
    return sum_loss / (batch_idx + 1)

############## validation関数 ##############
def val(model, val_loader, criterion, val_function, device):
    model.eval()
    
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, point) in enumerate(tqdm(val_loader, leave=False)):
            inputs = inputs.cuda(device, non_blocking=True)
            point = point[:, :, :, 2:].cuda(device, non_blocking=True)
            point = point.long()

            predicts, feature,coord = model(inputs)

            # Padding solution
            for detection_val in val_function:
                if batch_idx == 0:
                    for frame_idx in range(predicts.shape[2] // 2):
                        coordinate = detection_val.coordinater(predicts[:, :, frame_idx])
                        label_point = point[:, frame_idx]
                        detection_val.calculate(coordinate, label_point)
                else:
                    coordinate = detection_val.coordinater(predicts[:, :, 7])
                    label_point = point[:, 7]
                    detection_val.calculate(coordinate, label_point)
        
        
        # Padding solution
        for detection_val in val_function:
            for frame_idx in range(predicts.shape[2] // 2):
                coordinate = detection_val.coordinater(predicts[:, :, frame_idx + 8])
                label_point = point[:, frame_idx + 8]
                detection_val.calculate(coordinate, label_point)

        # Detection accuracy
        detection_accuracy = [detection_val() for detection_val in val_function]
        
    return detection_accuracy

@hydra.main(config_path="config", config_name='main_backbone_withcoord.yaml')
def main(cfg):
    print(f"epoch       :{cfg.parameter.epoch}")
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")

    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')
    
    # define accuracy class
    val_function = [Object_Detection(noise_strength=0.2, pool_range=11),
                    Object_Detection(noise_strength=0.3, pool_range=11),
                    Object_Detection(noise_strength=0.4, pool_range=11),
                    Object_Detection(noise_strength=0.5, pool_range=11),
                    Object_Detection(noise_strength=0.6, pool_range=11),
                    Object_Detection(noise_strength=0.7, pool_range=11)]
    
    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"

    with open(PATH_1, mode='w') as f:
        f.write("Epoch\tTrain Loss\n")
    
    PATH_acc = []
    for idx, val_fnc in enumerate(val_function):
        temp_PATH = f"result/accuracy{idx:04}.txt"
        PATH_acc.append(temp_PATH)
        with open(temp_PATH, mode='w') as f:
            for method_key,method_value in vars(val_fnc).items():
                f.write(f"{method_key}\t{method_value}\n")
            f.write("Epoch\tAccuracy\tPrecition\tRecall\tF1_Score\n")
    
    # dataset load
    train_loader, val_loader = dataload(cfg)

    move_limit = get_movelimit(train_loader.dataset, factor=1.1)

    # model setting
    if cfg.parameter.assignment:
        model = UNet_3D_FA(in_channels=1, n_classes=1,
                            feature_num=cfg.parameter.feature_num,
                            overlap_range=cfg.parameter.overlap_range,
                            noise_strength=cfg.parameter.noise_strength).cuda(device)
    else:
        model = UNet_3D(in_channels=1, n_classes=1,
                        noise_strength=cfg.parameter.noise_strength).cuda(device)

    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    criterion = Contrastive_Loss(overlap_range=move_limit)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max)  # Adam
    
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
            first_cycle_steps=cfg.scheduler.first,
            cycle_mult=cfg.scheduler.mult,
            max_lr=cfg.scheduler.max,
            min_lr=cfg.scheduler.min,
            warmup_steps=cfg.scheduler.warmup,
            gamma=cfg.scheduler.gamma)
    
    torch.backends.cudnn.benchmark = True

    best_acc = 0.
    best_method = 0
    for epoch in range(cfg.parameter.epoch):
        scheduler.step()

        # train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # validation
        val_accuracy = val(model, val_loader, criterion, val_function, device)

        ##### show result #####
        result_text = "Epoch{:3d}/{:3d} TrainLoss={:.5f}".format(
                        epoch + 1,
                        cfg.parameter.epoch,
                        train_loss)
        
        print(result_text)

        ##### write result #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
        for idx, accuracy in enumerate(val_accuracy):
            with open(PATH_acc[idx], mode='a') as f:
                f.write(f"{epoch+1}\t")
                for v in accuracy.values():
                    f.write(f"{v}\t")
                f.write("\n")
        
        np_acc = np.array([acc["Accuracy"] for acc in val_accuracy])
        np_acc_max = np.max(np_acc)
        np_acc_max_idx = np.argmax(np_acc)

        if np_acc_max >= best_acc:
            best_acc = np_acc_max
            best_method = np_acc_max_idx
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)

    print(f"best accuracy:{best_acc:.2%}")
    print("detection method")
    model.noise_strength = val_function[best_method].noise_strength
    for key, value in vars(val_function[best_method]).items():
        print(f"{key}:\t\t{value}\n")
    with open(PATH_acc[best_method], mode='a') as f:
        f.write(f"\nBest Method")

############## main ##############
if __name__ == '__main__':
    main()
