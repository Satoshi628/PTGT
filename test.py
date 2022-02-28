#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

#----- 自作ライブラリ -----#
from utils.dataset import OISTLoader, PTC_Loader, collate_delete_PAD, OISTLoader_add
import utils.transforms as tf
from utils.utils import get_movelimit
from utils.Transformer_to_track import Transformer_to_Track, Transformer_to_Track2, Transformer_to_Track3
from utils.evaluation import Object_Detection,Object_Tracking
from models.TATR import TATR_3D

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    
    test_transform = tf.Compose([])
    
    if cfg.dataset == "OIST":
        test_dataset = OISTLoader_add(root_dir=cfg.OIST.root_dir, Staning=cfg.OIST.Staning, mode="test", split=cfg.OIST.split, length=cfg.OIST.length, transform=test_transform)
        mode_limit_dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov", Staning=cfg.OIST.Staning, mode="train", split=cfg.OIST.split, length=cfg.OIST.length)
        move_limit = get_movelimit(mode_limit_dataset)
    if cfg.dataset == "PTC":
        test_dataset = PTC_Loader(root_dir=cfg.PTC.root_dir, Molecule=cfg.PTC.Molecule, Density=cfg.PTC.Density, mode="test", split=cfg.PTC.split, length=cfg.PTC.length, add_mode=True, transform=test_transform)
        mode_limit_dataset = PTC_Loader(root_dir=cfg.PTC.root_dir, Molecule=cfg.PTC.Molecule, Density="Mid", mode="train", split=cfg.PTC.split, length=cfg.PTC.length)
        move_limit = get_movelimit(mode_limit_dataset)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return test_loader, move_limit


############## validation関数 ##############
def test(model, test_loader, TtT, device):
    model.eval()
    coord_F = None

    tracking_fnc = Object_Tracking()
    detection_fnc = Object_Detection()

    with torch.no_grad():
        for batch_idx, (inputs, targets, point) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            inputs = inputs.cuda(device, non_blocking=True)
            point = point.cuda(device, non_blocking=True)
            point = point.long()

            vector, coord, coord_F = model.tracking_process(inputs, add_F_dict=coord_F)

            TtT.update(vector, coord)
            detection_fnc.calculate(coord.flatten(0, 1), point[:, :, :, 2:].flatten(0, 1))

        Detection_acc = detection_fnc()
        TtT()
        TtT.save_track()
        tracking_fnc.update(TtT.get_track(), test_loader.dataset.CP_data)
        Tracking_acc = tracking_fnc()

    return Tracking_acc, Detection_acc

@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    print(f"dataset     :{cfg.dataset}")


    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH = "result/test_Mid.txt"

    with open(PATH, mode='w') as f:
        f.write("")
    
    # dataset load
    test_loader, move_limit = dataload(cfg)

    # define model
    model = TATR_3D(back_bone_path=cfg.parameter.back_bone,
                    noise_strength=cfg.parameter.noise_strength,
                    pos_mode=cfg.parameter.pos,
                    assignment=cfg.parameter.assignment,
                    feature_num=cfg.parameter.feature_num,
                    overlap_range=cfg.parameter.overlap_range,
                    encode_mode=cfg.parameter.encoder,
                    move_limit=move_limit[0]).cuda(device)

    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    torch.backends.cudnn.benchmark = True


    #TtT = Transformer_to_Track(move_limit=move_limit, connect_method="N")
    #TtT = Transformer_to_Track2(connect_method="N", move_limit=move_limit)
    TtT = Transformer_to_Track3(tracking_mode="P", move_limit=move_limit)

    Tracking, Detection = test(model, test_loader, TtT, device)

    with open(PATH, mode='a') as f:
        for k in Detection.keys():
            f.write(f"{k}\t")
        f.write("\n")
        for v in Detection.values():
            f.write(f"{v}\t")
        f.write("\n")
        for k in Tracking.keys():
            f.write(f"{k}\t")
        f.write("\n")
        for v in Tracking.values():
            f.write(f"{v}\t")
    
    #Accuracy, Precition, Recall, F1_Score
    print("Detection Evalution")
    for k, v in Detection.items():
        print(f"{k}".ljust(20) + f":{v}")

    print("Tracking Evalution")
    for k, v in Tracking.items():
        print(f"{k}".ljust(20) + f":{v}")



############## main ##############
if __name__ == '__main__':
    main()
