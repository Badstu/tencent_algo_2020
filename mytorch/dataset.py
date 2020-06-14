import data
import os
import sys
import re
import matplotlib
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from gensim.models import word2vec, keyedvectors
import logging

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader


def collate_fn(data):
    data.sort(key = lambda x: x[0], reverse=True)
    t = [x[0] for x in data]
    embed_features = [x[1] for x in data]
    age = [x[2] for x in data]
    gender = [x[3] for x in data]

    embed_features = rnn_utils.pad_sequence(embed_features, batch_first=True, padding_value=0)
    embed_features = rnn_utils.pack_padded_sequence(embed_features, lengths=t, batch_first=True)
    return t, embed_features, age, gender


class RecordDataset(Dataset):
    def __init__(self, list_grouped, creative_model, ad_model, product_model, advertiser_model, industry_model, data_type="train"):
        self.creative_model = creative_model
        self.ad_model = ad_model
        self.product_model = product_model
        self.advertiser_model = advertiser_model
        self.industry_model = industry_model
        
        self.list_grouped = list_grouped
        self.data_type = data_type
    
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        record = self.list_grouped.iloc[index, :]
        user_id = record["user_id"]
        t = len(record["ad_id"])
        if self.data_type == "train" or self.data_type == "valid":
            age = record["age"]
            gender = record["gender"]
        elif self.data_type == "test":
            pass
        
        # ad_embedding
        ad_embedding = self.get_embedding_from_grouped(user_id, record, column_name="ad_id")
        #creative_embedding
        creative_embedding = self.get_embedding_from_grouped(user_id, record, column_name="creative_id")
        #product_embedding
        product_embedding = self.get_embedding_from_grouped(user_id, record, column_name="product_id")
        #advertiser_embedding
        advertiser_embedding = self.get_embedding_from_grouped(user_id, record, column_name="advertiser_id")
        #industry_embedding
        industry_embedding = self.get_embedding_from_grouped(user_id, record, column_name="industry")
        
        embed_features = torch.cat([ad_embedding, creative_embedding, product_embedding, advertiser_embedding, industry_embedding], dim=1)
            
        return t, embed_features, age, gender
        
    def __len__(self):
        return len(self.list_grouped)
    
    
    def get_embedding_from_grouped(self, user_id, record, column_name):
        if column_name == "ad_id":
            model = self.ad_model
        elif column_name == "creative_id":
            model = self.creative_model
        elif column_name == "industry":
            model = self.industry_model
        elif column_name == "product_id":
            model = self.product_model
        elif column_name == "advertiser_id":
            model = self.advertiser_model

        if column_name == "industry":
            embedding = [np.zeros(100, ) if pd.isnull(x) else model[str(int(x))] for x in record[column_name]]
        elif column_name == "product_id":
            embedding = [np.zeros(200, ) if pd.isnull(x) else model[str(int(x))] for x in record[column_name]]
        else:
            embedding = [model[str(x)] for x in record[column_name]]
        
        embedding = torch.Tensor(embedding)
        
        return embedding
