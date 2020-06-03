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

creative_model = keyedvectors.KeyedVectors.load_word2vec_format("checkpoints/creative_model.w2v", binary=True)
ad_model = keyedvectors.KeyedVectors.load_word2vec_format("checkpoints/ad_model.w2v", binary=True)
product_model = keyedvectors.KeyedVectors.load_word2vec_format("checkpoints/product_model.w2v", binary=True)
advertiser_model = keyedvectors.KeyedVectors.load_word2vec_format("checkpoints/advertiser_model.w2v", binary=True)
industry_model = keyedvectors.KeyedVectors.load_word2vec_format("checkpoints/industry_model.w2v", binary=True)

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


def total_embed(grouped):
    final_embedding = pd.DataFrame(np.zeros(800, )).T
    final_embedding.insert(0, "user_id", 0)
    flag = 0
    for user_id, records in tqdm(grouped):
        records = records.sort_values(by="time")

        # ad_embedding
        ad_embedding = get_embedding_from_grouped(user_id, records, column_name="ad_id")
        #creative_embedding
        creative_embedding = get_embedding_from_grouped(user_id, records, column_name="creative_id")
        #product_embedding
        product_embedding = get_embedding_from_grouped(user_id, records, column_name="product_id")
        '''
        #advertiser_embedding
        advertiser_embedding = get_embedding_from_grouped(user_id, records, column_name="advertiser_id")
        #industry_embedding
        industry_embedding = get_embedding_from_grouped(user_id, records, column_name="industry")
        '''

        embed_features = np.concatenate([ad_embedding, creative_embedding, product_embedding])
        embed_features = pd.DataFrame([embed_features])
        embed_features.insert(0, "user_id", user_id)

        final_embedding = final_embedding.append(embed_features)
        
#         flag += 1
#         if flag > 10:
#             break
    return final_embedding

train_embedding = total_embed(train_grouped)
test_embedding = total_embed(test_grouped)

train_embedding.to_csv("checkpoints/train_embedding.csv", index=False)
test_embedding.to_csv("checkpoints/test_embedding.csv", index=False)
