import numpy as np
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
    msk_name = img_path.split("/")[-1].replace(".jpg", "_mask.png")
    cv2.imwrite(CFG.MSK_PATH+f"/{msk_name}", mask,)


def make_dataset_multi():
    img_list = glob.glob(CFG.DATASET_PATH+"/*.jpg")
    with futures.ProcessPoolExecutor(max_workers=10) as e:
        e.map(make_dataset, img_list)



if __name__ ==  "__main__":
    make_dataset_multi()