import ShallowLearn.ImageHelper as ih
import matplotlib.pyplot as plt 
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
path = "/media/ziad/Expansion/Cloud_Masks/"

masks = os.listdir(path)
masks = [i for i in masks if i.endswith(".SAFE")]

df = pd.DataFrame()
# implement tdqm in loop below
pbar = tqdm(total = len(masks), position= 0, leave = True)

for mask in masks:
    data = ih.load_img(path + mask).reshape(-1)
    hist, bin_edges = np.histogram(data, bins = 100)
    name = {
        "Name": mask, 
        "Mean": data.mean(), 
        "Std": data.std(), 
        "Median": np.median(data), 
        "Min": data.min(), 
        "Max": data.max(),
        "hist": [hist],
        "bin_edges": [bin_edges]
    }
    df = pd.concat([df, pd.DataFrame(name, index = [0])])
    pbar.update(1)

df.to_csv("Data/Cloud_Mask_Stats.csv")
pbar.close()