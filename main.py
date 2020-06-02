# -*- coding:utf-8 -*-

import data
import lightgbm as lgb
import numpy as np
import os
import sys
import pandas as pd
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
train_user, valid_user = train_test_split(train_user, test_size=0.33, random_state=42)
train_record = data.get_part_click(train_click, train_user)
valid_record = data.get_part_click(train_click, valid_user)

# train_record
train_record = pd.merge(train_record, train_ad, on="creative_id")
# valid_record
valid_record = pd.merge(valid_record, train_ad, on="creative_id")

train_features, train_age, train_gender = data.split_feature_target(train_record, keep_user=True)
valid_features, valid_age, valid_gender = data.split_feature_target(train_record, keep_user=True)

column_names = ["creative_id", "ad_id", "product_id", "advertiser_id", "industry"]
w2v_models = [creative_model, ad_model, product_model, advertiser_model, industry_model]
for column_name, w2v_model in zip(column_names, w2v_models):
    print(column_name, "START")
    if column_name == "industry":
        embedding_df = train_features[column_name].apply(lambda x: np.zeros(100, ) if pd.isnull(x) else w2v_model[str(int(x))]).apply(pd.Series)
    elif column_name == "product_id":
        embedding_df = train_features[column_name].apply(lambda x: np.zeros(200, ) if pd.isnull(x) else w2v_model[str(int(x))]).apply(pd.Series)
    else:
        embedding_df = train_features[column_name].apply(lambda x: w2v_model[str(x)]).apply(pd.Series)
    train_features = pd.concat([train_features, embedding_df], axis=1).drop(column_name, axis=1)
    print(column_name, "FINISH")
    
train_features.to_csv("main_features.csv", index=False)
print("finish saving csv")