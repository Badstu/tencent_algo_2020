import data
import lightgbm as lgb
import numpy as np
import os
import sys
import re
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


print("START loading train embedding and train user info")
train_np = np.loadtxt("embed/train/train_embedding_800_1.csv", delimiter=", ")
train_np[train_np == 0] = np.nan

train_root = "dataset/train/"
train_user_path = os.path.join(train_root, "user.csv")
train_user = pd.read_csv(train_user_path, index_col="user_id")
print("FINISH load train_np, train_user")
print("===========================================================================")

print("START get train_features, train_age, train_gender, and random split train/valid data")
uid = train_np[:, 0].astype(int)
train_age = train_user.loc[uid, "age"]
train_gender = train_user.loc[uid, "gender"]

train_features = train_np[:, 1:201]
train_age = train_age.values - 1
train_gender = train_gender.values - 1

train_features, valid_features,\
train_age, valid_age,\
train_gender, valid_gender = train_test_split(train_features,\
                                              train_age,\
                                              train_gender,\
                                              test_size=0.33,\
                                              random_state=42)
print("FINISH random split train/valid data")
print("===========================================================================")

print("START construct lgb train valid data")
lgb_traindata_gender = lgb.Dataset(train_features, train_gender)
lgb_traindata_age = lgb.Dataset(train_features, train_age)

lgb_valdata_gender = lgb.Dataset(valid_features, valid_gender, reference=lgb_traindata_gender)
lgb_valdata_age = lgb.Dataset(valid_features, valid_age, reference=lgb_traindata_age)
print("FINISH construct lgb train valid data")
print("===========================================================================")


print("START train model")
# TODO 性别模型的训练
gender_model = lgb_model(model_kind="gender")
gender_model.train(lgb_traindata_gender, lgb_valdata_gender)
gender_model.save_model()

# TODO 年龄模型的训练
age_model = lgb_model(model_kind="age")
age_model.train(lgb_traindata_age, lgb_valdata_age)
age_model.save_model()
print("FINISH train model and save model")
print("===========================================================================")


'''
# 导入已保存模型
gender_model = lgb_model(model_kind="gender")
gender_model.load_model()
age_model = lgb_model(model_kind="age")
age_model.load_model()
'''

print("START valid acc of predict")
# TODO 性别模型的预测
valid_gender_predict = gender_model.predict(valid_features)
valid_gender_predict = gender_model.transform_pred(valid_gender_predict)
acc_gender = accuracy_score(valid_gender_predict, valid_gender)

# TODO 年龄模型的预测
valid_age_predict = age_model.predict(valid_features)
valid_age_predict = age_model.transform_pred(valid_age_predict)
acc_age = accuracy_score(np.array(valid_age_predict), valid_age)

print("In valid data, accuracy of gender is {}, accuracy of age is {}".format(acc_gender, acc_age))
print("FINISH")
print("===========================================================================")

print("START test predict")
test_np = np.loadtxt("embed/test/test_embedding_800_1.csv", delimiter=", ")
test_np[test_np == 0] = np.nan
test_uid = test_np[:, 0].astype(int)
test_features = test_np[:, 1:201]

# TODO 性别模型的预测
test_gender_predict = gender_model.predict(test_features)
test_gender_predict = gender_model.transform_pred(test_gender_predict)
# TODO 年龄模型的预测
test_age_predict = age_model.predict(test_features)
test_age_predict = age_model.transform_pred(test_age_predict)

result = pd.DataFrame({"user_id": test_uid, "predicted_age": test_age_predict, "predicted_gender": test_gender_predict})
result.to_csv("results.csv", index=False)

print("FINISH ALL and save result to results.csv")
print("===========================================================================")