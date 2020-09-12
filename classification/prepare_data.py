import os 
from glob import glob 
import pandas as pd 
import linecache

categories = [name for name in os.listdir("../data/text") if os.path.isdir('../data/text/' + name)]

datasets = pd.DataFrame(columns=["title", "category"])

for cat in categories:
    path = "../data/text/" + cat + "/*.txt"
    files = glob(path)
    for file in files:
        title = linecache.getline(file,3)
        row = pd.Series([title, cat], index=datasets.columns)
        datasets = datasets.append(row, ignore_index=True)

datasets = datasets.sample(frac=1, random_state=0).reset_index(drop=True)

