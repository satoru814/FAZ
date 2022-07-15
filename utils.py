import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
import cv2
import json
from concurrent import futures
import torch
from config import CFG
from shapely.geometry import Polygon
from shapely.geometry import Point

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def make_dataset(img_path):
    img = cv2.imread(img_path)
    
    with open(img_path.replace(".jpg", ".json"), "r") as f:
        jso = json.load(f)

    polygon_mask = Polygon(jso["shapes"][0]["points"])
    mask = np.zeros((img.shape[0], img.shape[1]))
    for h in range(mask.shape[0]):
        for w in range(mask.shape[1]):
            b = polygon_mask.contains(Point(w,h))
            if b :
                mask[h,w] = 1
    msk_name = img_path.replace(".jpg", "_mask.png")
    cv2.imwrite(msk_name, mask)


def make_dataset_multi():
    img_list = glob.glob(CFG.DATASET_PATH+"/*.jpg")
    with futures.ProcessPoolExecutor(max_workers=50) as e:
        e.map(make_dataset, img_list)


def make_dataset_df():
    img_paths = []
    msk_paths = []
    img_list = glob.glob(CFG.DATASET_PATH+"/*.jpg")
    for img_path in img_list:
        msk_path = img_path.replace(".jpg", "_mask.png")
        img_paths.append(img_path)
        msk_paths.append(msk_path)

    df = pd.DataFrame({"img_paths":img_paths, "msk_paths":msk_paths})
    df.to_csv(CFG.DF_PATH, index=None)

    return None


class FAZ_Dataset(Dataset):
    def __init__(self, df_path, transform=None, is_train=True):
        self.df = pd.read_csv(df_path)
        self.transform = transform
        self.img_list = self.df["img_paths"].values
        self.msk_list = self.df["msk_paths"].values
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        msk_path = self.msk_list[idx]
        img = cv2.imread(img_path)
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            transformed = self.transform(image=img, mask=msk)
            img = transformed["image"]
            msk = transformed["mask"]
            # print(torch.mean(msk))
        return img, msk
    def __len__(self):
        return len(self.df)


def get_transform():
    transform = A.Compose([
        A.augmentations.geometric.resize.Resize(CFG.IMGSIZE, CFG.IMGSIZE),
        A.HorizontalFlip(0.5),
        ToTensorV2(),
    ])
    return transform

def calc_IOU(preds, msks, THRESHOLD=CFG.THRESHOLD):
    iou = 0
    for i in range(preds.shape[0]):
        pred = preds[i]
        msk = msks[i]
        pred = (pred > THRESHOLD)
        intersect = np.logical_and(pred, msk)
        union = np.logical_or(pred, msk)
        iou +=  np.sum(intersect) / np.sum(union)
    iou /= preds.shape[0]
    return iou


def save_figure(pred, msk, img, epoch):
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(img)
    ax[1].imshow(msk)
    ax[2].imshow(pred)
    plt.savefig(CFG.SAVE_PATH+f"/{epoch}.png")
    return None

if __name__ ==  "__main__":
    make_dataset_multi()