# -*- coding: utf-8 -*-

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all" 

import data
import lightgbm as lgb
import numpy as np
import os
import sys
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from gensim.models import word2vec
import logging

from model import lgb_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, grouped_record, column_name):
        self.grouped_record = grouped_record
        self.column_name = column_name
        
    def __iter__(self):
        for user_id, record in grouped_record:
            record = record.sort_values(by="time")
            if self.column_name == "product_id" or self.column_name == "industry":
                p_id = record[self.column_name]
                p_id = p_id[~pd.isnull(p_id)].astype("int")
                sentence = list(map(str, list(p_id)))
            else:
                sentence = list(map(str, list(record[self.column_name])))
            yield sentence


train_ad, train_click, train_user, test_ad, test_click = data.load_data()
train_record = pd.merge(train_click, train_ad, on="creative_id")
test_record = pd.merge(test_click, test_ad, on="creative_id")
all_record = pd.concat([train_record, test_record])

grouped_record = all_record.groupby("user_id")

creative_sens = MySentences(grouped_record, "creative_id")
ad_sens = MySentences(grouped_record, "ad_id")
product_sens = MySentences(grouped_record, "product_id")
advertiser_sens = MySentences(grouped_record, "advertiser_id")
industry_sens = MySentences(grouped_record, "industry")

creative_model = word2vec.Word2Vec(creative_sens, min_count=3, size=200, workers=4, sg=1, window=10)
creative_model.wv.save_word2vec_format("checkpoints/creative_model_win10.w2v", binary=True)

ad_model = word2vec.Word2Vec(ad_sens, min_count=1, size=200, workers=4, sg=1, window=10)
ad_model.wv.save_word2vec_format("checkpoints/ad_model_win10.w2v", binary=True)

product_model = word2vec.Word2Vec(product_sens, min_count=3, size=200, workers=4, sg=1, window=10)
product_model.wv.save_word2vec_format("checkpoints/product_model_win10.w2v", binary=True)

advertiser_model = word2vec.Word2Vec(advertiser_sens, min_count=3, size=100, workers=4, sg=1, window=10)
advertiser_model.wv.save_word2vec_format("checkpoints/advertiser_model_win10.w2v", binary=True)

industry_model = word2vec.Word2Vec(industry_sens, min_count=3, size=100, workers=4, sg=1, window=10)
industry_model.wv.save_word2vec_format("checkpoints/industry_model_win10.w2v", binary=True)