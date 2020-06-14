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
train_np = np.loadtxt("embed/train/ag_train_embedding_800_1.csv", delimiter=", ")
train_np[train_np == 0] = np.nan

# train_root = "dataset/train/"
# train_user_path = os.path.join(train_root, "user.csv")
# train_user = pd.read_csv(train_user_path, index_col="user_id")
print("FINISH load train_np, train_user")
print("===========================================================================")

tc_train_age = pd.read_csv("tc/train_target_age.csv")
tc_train_gender = pd.read_csv("tc/train_target_gender.csv")

print("START get train_features, train_age, train_gender, and random split train/valid data")
uid = train_np[:, 0].astype(int)
train_age = train_np[:, -2].astype(int)
train_gender = train_np[:, -1].astype(int)

train_features = train_np[:, 1:-2]
train_features_gender = np.concatenate([train_features, tc_train_gender.values[:, 1:]], axis=1)
train_features_age = np.concatenate([train_features, tc_train_age.values[:, 1:]], axis=1)

train_age = train_age - 1
train_gender = train_gender - 1

train_features_gender, valid_features_gender,\
train_features_age, valid_features_age,\
train_age, valid_age,\
train_gender, valid_gender = train_test_split(train_features_gender,\
                                              train_features_age,\
                                              train_age,\
                                              train_gender,\
                                              test_size=0.33,\
                                              random_state=42)

print("FINISH random split train/valid data")
print("===========================================================================")

print("START construct lgb train valid data")
lgb_traindata_gender = lgb.Dataset(train_features_gender, train_gender)
lgb_traindata_age = lgb.Dataset(train_features_age, train_age)

lgb_valdata_gender = lgb.Dataset(valid_features_gender, valid_gender, reference=lgb_traindata_gender)
lgb_valdata_age = lgb.Dataset(valid_features_age, valid_age, reference=lgb_traindata_age)
print("FINISH construct lgb train valid data")
print("===========================================================================")


print("START train model")
# TODO 性别模型的训练
gender_model = lgb_model(model_kind="gender")
gender_model.train(lgb_traindata_gender, lgb_valdata_gender)
gender_model.save_model("checkpoints/gender_model_complex.pkl")

# TODO 年龄模型的训练
age_model = lgb_model(model_kind="age")
age_model.train(lgb_traindata_age, lgb_valdata_age)
age_model.save_model("checkpoints/age_model_complex.pkl")
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
valid_gender_predict = gender_model.predict(valid_features_gender)
valid_gender_predict = gender_model.transform_pred(valid_gender_predict)
acc_gender = accuracy_score(valid_gender_predict, valid_gender)

# TODO 年龄模型的预测
valid_age_predict = age_model.predict(valid_features_age)
valid_age_predict = age_model.transform_pred(valid_age_predict)
acc_age = accuracy_score(np.array(valid_age_predict), valid_age)

print("In valid data, accuracy of gender is {}, accuracy of age is {}".format(acc_gender, acc_age))
print("FINISH")
print("===========================================================================")

'''
# 测试集预测
print("START loading test embedding")
test_np = np.loadtxt("embed/test/ag_test_embedding_800_1.csv", delimiter=", ")
test_np[test_np == 0] = np.nan

tc_test_age = pd.read_csv("tc/test_target_age.csv")
tc_test_gender = pd.read_csv("tc/test_target_gender.csv")
print("===========================================================================")

print("START test predict")
test_uid = test_np[:, 0].astype(int)
test_features = test_np[:, 1:]

test_features_gender = np.concatenate([test_features, tc_test_gender.values[:, 1:]], axis=1)
test_features_age = np.concatenate([test_features, tc_test_age.values[:, 1:]], axis=1)

# TODO 性别模型的预测
test_gender_predict = gender_model.predict(test_features_gender)
test_gender_predict = gender_model.transform_pred(test_gender_predict)
# TODO 年龄模型的预测
test_age_predict = age_model.predict(test_features_age)
test_age_predict = age_model.transform_pred(test_age_predict)

result = pd.DataFrame({"user_id": test_uid, "predicted_age": test_age_predict, "predicted_gender": test_gender_predict})
result.loc[:, "predicted_age"] += 1
result.loc[:, "predicted_gender"] += 1
result.to_csv("results_complex.csv", index=False)

print("FINISH ALL and save result to results.csv")
print("===========================================================================")
'''