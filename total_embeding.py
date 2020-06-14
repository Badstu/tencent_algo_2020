# -*- coding:utf-8 -*-

import data
import lightgbm as lgb
import numpy as np
import os
import sys
import pandas as pd
import matplotlib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from gensim.models import word2vec, keyedvectors
import logging

from model import lgb_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


suffix = "_win10"
creative_model = keyedvectors.KeyedVectors.load_word2vec_format("checkpoints/creative_model{}.w2v".format(suffix), binary=True)
ad_model = keyedvectors.KeyedVectors.load_word2vec_format("checkpoints/ad_model{}.w2v".format(suffix), binary=True)
product_model = keyedvectors.KeyedVectors.load_word2vec_format("checkpoints/product_model{}.w2v".format(suffix), binary=True)
advertiser_model = keyedvectors.KeyedVectors.load_word2vec_format("checkpoints/advertiser_model{}.w2v".format(suffix), binary=True)
industry_model = keyedvectors.KeyedVectors.load_word2vec_format("checkpoints/industry_model{}.w2v".format(suffix), binary=True)

train_ad, train_click, train_user, test_ad, test_click = data.load_data()
# train_record
train_record = pd.merge(train_click, train_ad, on="creative_id")
# test_record
test_record = pd.merge(test_click, test_ad, on="creative_id")

# TODO train embedding
train_grouped = train_record.groupby("user_id")
test_grouped = test_record.groupby("user_id")

def get_embedding_from_grouped(user_id, records, column_name, keep_uid=False):
    if column_name == "ad_id":
        model = ad_model
    elif column_name == "creative_id":
        model = creative_model
    elif column_name == "industry":
        model = industry_model
    elif column_name == "product_id":
        model = product_model
    elif column_name == "advertiser_id":
        model = advertiser_model
    
    if column_name == "industry":
        embedding = records[column_name].apply(lambda x: np.zeros(100, ) if pd.isnull(x) else model[str(int(x))]).apply(pd.Series)
    elif column_name == "product_id":
        embedding = records[column_name].apply(lambda x: np.zeros(200, ) if pd.isnull(x) else model[str(int(x))]).apply(pd.Series)
    else:
        embedding = records[column_name].apply(lambda x: model[str(x)]).apply(pd.Series)
    embedding = embedding.mean()
    
    if keep_uid:
        embedding.insert(0, "user_id", user_id)
    return embedding


def total_embed(grouped, data_type="train"):
#     id = 1
#     flag = 0
    if data_type == "train":
        f = open("embed/train/ag_train_embedding_800{}.csv".format(suffix), "w")
    else:
        f = open("embed/test/ag_test_embedding_800{}.csv".format(suffix), "w")
    for user_id, records in tqdm(grouped):
        records = records.sort_values(by="time")

        # ad_embedding
        ad_embedding = get_embedding_from_grouped(user_id, records, column_name="ad_id")
        #creative_embedding
        creative_embedding = get_embedding_from_grouped(user_id, records, column_name="creative_id")
        #product_embedding
        product_embedding = get_embedding_from_grouped(user_id, records, column_name="product_id")
        #advertiser_embedding
        advertiser_embedding = get_embedding_from_grouped(user_id, records, column_name="advertiser_id")
        #industry_embedding
        industry_embedding = get_embedding_from_grouped(user_id, records, column_name="industry")

        embed_features = np.concatenate([ad_embedding, creative_embedding, product_embedding, advertiser_embedding, industry_embedding])
        
        if data_type == "train":
            age = train_user[train_user["user_id"] == user_id].loc[:, "age"].values[0]
            gender = train_user[train_user["user_id"] == user_id].loc[:, "gender"].values[0]
            s_to_f = str(user_id) + ', ' + str(list(embed_features))[1:-1] + ', ' + str(age) + ', ' + str(gender) + '\n'
        elif data_type == "test":
            s_to_f = str(user_id) + ', ' + str(list(embed_features))[1:-1] + '\n'
            
        f.write(s_to_f)

#         flag += 1
#         if flag % 45000 == 0:
#             f.close()
#             id += 1
#             if data_type == "train":
#                 f = open("embed/train/train_embedding{}.csv".format(id), "w")
#             else:
#                 f = open("embed/test/test_embedding{}.csv".format(id), "w")
    f.close()

total_embed(train_grouped, data_type="train")
total_embed(test_grouped, data_type="test")
