import json
import numpy as np
import random
import pandas as pd
import csv
from collections import Counter
import matplotlib.pyplot as plt
import pickle

small_user_item = {}
small_user_item_rating = {}
with open("ml-100k/ratings.csv", "rb") as infile:
    reader = csv.reader(infile)
    next(reader, None)  # skip the headers
    for row in reader:

        u = int(row[0])
        b = int(row[1])
        r = float(row[2])
        try:
            small_user_item[u].append(b)
        except:
            small_user_item[u] = [b]
        small_user_item_rating[(u,b)] = r

all_movie = []
for elem in small_user_item:
    all_movie += small_user_item[elem]
all_movie = list(set(all_movie))

movie_id_to_index = {}
for elem in all_movie:
    if elem not in movie_id_to_index:
        movie_id_to_index[elem] = len(movie_id_to_index)

movie_index_to_id = {k:v for v,k in movie_id_to_index.items()}

TRAIN = {}
VALIDATE = {}
TEST = {}

for elem in small_user_item:
    total_len = len(small_user_item[elem])
    a = small_user_item[elem]
    random.shuffle(a)
    tr = a[:int(round(total_len*0.6))]
    va = a[int(round(total_len*0.6)):int(round(total_len*0.8))]
    te = a[int(round(total_len*0.8)):]
    TRAIN[elem] = tr
    VALIDATE[elem] = va
    TEST[elem] = te

TRAIN_list = []
VALI_list = []
TEST_list = []

for u in TRAIN:
    for b in TRAIN[u]:
        if small_user_item_rating[(u,b)] < 3:
            TRAIN_list.append((u,movie_id_to_index[b],small_user_item_rating[(u,b)],0))
        elif small_user_item_rating[(u,b)] >= 3:
            TRAIN_list.append((u,movie_id_to_index[b],small_user_item_rating[(u,b)],1))
            
for u in VALIDATE:
    for b in VALIDATE[u]:
        if small_user_item_rating[(u,b)] < 3:
            VALI_list.append((u,movie_id_to_index[b],small_user_item_rating[(u,b)],0))
        elif small_user_item_rating[(u,b)] >= 3:
            VALI_list.append((u,movie_id_to_index[b],small_user_item_rating[(u,b)],1))
            
for u in TEST:
    for b in TEST[u]:
        if small_user_item_rating[(u,b)] < 3:
            TEST_list.append((u,movie_id_to_index[b],small_user_item_rating[(u,b)],0))
        elif small_user_item_rating[(u,b)] >= 3:
            TEST_list.append((u,movie_id_to_index[b],small_user_item_rating[(u,b)],1))



with open("ml-100k/ml-100k_Train_list.txt", "wb") as fp:  
    pickle.dump(TRAIN_list, fp)
          
with open("ml-100k/ml-100k_Vali_list.txt", "wb") as fp:  
    pickle.dump(VALI_list, fp)
          
with open("ml-100k/ml-100k_Test_list.txt", "wb") as fp:  
    pickle.dump(TEST_list, fp)

with open("ml-100k/Movie_index_to_id.json", 'w') as f:
    json.dump(movie_index_to_id,f)















