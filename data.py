import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score

def load_data():
    train_root = "dataset/train/"
    train_user_path = os.path.join(train_root, "user.csv")
    train_ad_path = os.path.join(train_root, "ad.csv")
    train_click_path = os.path.join(train_root, "click_log.csv")
    test_root = "dataset/test/"
    test_ad_path = os.path.join(test_root, "ad.csv")
    test_click_path = os.path.join(test_root, "click_log.csv")
    
    train_ad = pd.read_csv(train_ad_path, na_values="\\N")
    train_click = pd.read_csv(train_click_path, na_values="\\N")
    train_user = pd.read_csv(train_user_path, na_values="\\N")
    test_ad = pd.read_csv(test_ad_path, na_values="\\N")
    test_click = pd.read_csv(test_click_path, na_values="\\N")
    
    return train_ad, train_click, train_user, test_ad, test_click


def get_part_click(total_click, list_user_id, on="user_id"):
    part_record = pd.merge(total_click, list_user_id, on=on)
    return part_record


def get_ad_inform(creative_id, data_ad):
    ad_inform = data_ad[data_ad["creative_id"] == creative_id]
#     print(ad_inform.astype(int))
    return ad_inform.astype(int)


def split_feature_target(raw_features, keep_user=False):
    if keep_user == True:
        train_features = raw_features.iloc[:, [0, 1, 2, 3, 6, 7, 8, 9, 10]]
        train_age = raw_features.iloc[:, [1, 4]]
        train_gender = raw_features.iloc[:, [1, 5]]
    else:
        train_features = raw_features.iloc[:, [0, 2, 3, 6, 7, 8, 9, 10]]
        train_age = raw_features.iloc[:, 4]
        train_gender = raw_features.iloc[:, 5]
    
    return train_features, train_age, train_gender

def measure_unique_user(record_pred, data_record, data_user, column_name="gender"):
    df_pred = pd.DataFrame(data_record.user_id)
    df_pred[column_name] = np.array(record_pred)
    
    uni_user_pred = df_pred.groupby("user_id").agg({column_name: lambda x: x.value_counts().index[0]})
    pred = uni_user_pred.iloc[:, 0].values + 1
    target = data_user.sort_values("user_id")[column_name].values
    acc_score = accuracy_score(pred, target)
    
    return uni_user_pred, acc_score

def word_embedding(train_features, w2v_model, column_name):
    embedding_df = train_features[column_name].apply(lambda x: w2v_model[str(x)]).apply(pd.Series)
    new_train_features = pd.concat([train_features, embedding_df], axis=1).drop(column_name, axis=1)
    return new_train_features