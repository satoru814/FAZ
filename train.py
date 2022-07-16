
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import utils
from models import Unet, NestedUnet, LWBNAUnet


import numpy as np
import argparse
import time
import os
import sys
from itertools import chain
import matplotlib as plt
from PIL import Image
import numpy as np
import wandb
from config import CFG
import random
import argparse


def set_requires_grad(models, requires=False):
    if not isinstance(models, list):
        models = [models]
    for model in models:
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires


def main():
    #argparse
    parser = argparse.ArgumentParser(description='FAZ')
    parser.add_argument("--wandb", "-w", action="store_true", default=False, help="True -> wandb log is on ")
    args = parser.parse_args()


    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    
    in_channel, out_channel = CFG.Unet.in_channel, CFG.Unet.out_channel
    Net = LWBNAUnet.LWBNAUnet(in_channel, out_channel).to(device)

    #set loss functions
    l1_loss = nn.L1Loss().to(device)

    #set optimizer
    optimizer = torch.optim.Adam(Net.parameters(), **CFG.Unet.optimizer)

    #dataset
    trans = utils.get_transform()
    train_dataset =  utils.FAZ_Dataset(CFG.DF_PATH,is_train=True, transform=trans)
    train_loader = DataLoader(train_dataset, **CFG.Unet.dataloader)

    #wandb
    try:
        api_key = user_secrets.get_secret("WANDB")
        wandb.login(CFG.wandb_key)
        anonymous = None
    except:
        anonymous = "must"
    if args.wandb:
        run = wandb.init(**CFG.wandb, settings=wandb.Settings(code_dir="."))
        wandb.run.log_code(".")
        wandb.watch(models=(Net), log_freq=1000)


    print("data_size:",len(train_loader))
    for epoch in range(CFG.Unet.epoch):
        Net.train()
        losses = {"L1_loss" : 0, "iou":0}
        iter = 0
        print("epoch",epoch)
        for i , data in enumerate(train_loader):
            optimizer.zero_grad()

            img, msk = data
            img = img.to(device).float()
            msk = msk.to(device)
            msk = msk.view(-1, 1, CFG.IMGSIZE, CFG.IMGSIZE).float()
            pred = Net(img)
            loss =  l1_loss(pred, msk)

            loss.backward()
            optimizer.step()

            #loss
            losses["L1_loss"] += loss.item()

            #metric calc
            img = img.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            msk = msk.detach().cpu().numpy()

            iou = utils.calc_IOU(pred, msk)
            losses["iou"] += iou

            iter += 1
        

        pred = pred[0][0]
        msk = msk[0][0]
        img = img[0].transpose(1,2,0)
        utils.save_figure(pred, msk, img, epoch)

        for k,v in losses.items():
            losses[k] = v/iter

        if args.wandb:
            wandb.log(losses)

    torch.save(Net.state_dict(), CFG.MODEL_SAVE_PATH)
    run.finish()


if __name__ == "__main__":
    main()
