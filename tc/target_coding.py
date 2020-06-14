import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

datapath = "dataset/train/"

train_ad = pd.read_csv(datapath + "ad.csv", na_values="\\N")
train_click = pd.read_csv(datapath + "click_log.csv", na_values="\\N")
train_user = pd.read_csv(datapath + "user.csv", na_values="\\N")
train_data = train_click.merge(train_ad, how="left", on="creative_id", )
train_data= train_data.merge(train_user, how="left", on="user_id", )
train_data.fillna(0,inplace=True)
train_data[['creative_id','ad_id','product_id','advertiser_id','industry']] = train_data[['creative_id','ad_id','product_id','advertiser_id','industry']].astype('object')

# 目标编码
def target_encode(X, cols, target_feature):
    X_ = pd.DataFrame()
    X_['user_id']= X['user_id']
    for col in tqdm(cols):
        print('Target Encoding: {}'.format(col))
        grouped=X.groupby([col])[target_feature]
        X_[col+'_target_encoded_mean'] = X[col].map(dict(grouped.mean()))
        X_[col+'_target_encoded_median'] = X[col].map(dict(grouped.median()))
        X_[col+'_target_encoded_std'] = X[col].map(dict(grouped.std()))
    return X_

feature = ['creative_id','ad_id','product_id','advertiser_id','industry']
target_input = train_data[feature+['age','gender','user_id']]

# 目标编码gender
train_target_gender = target_encode(target_input,feature,'gender')
train_target_gender = train_target_gender.groupby('user_id').agg('median')

# 目标编码age
train_target_age = target_encode(target_input,feature,'age')
train_target_age  = train_target_age.groupby('user_id').agg('median')

train_target_gender.to_csv("tc/train_target_gender.csv")
train_target_age.to_csv("tc/train_target_age.csv")

# K折目标编码
kf = KFold(n_splits = 8, shuffle = False, random_state=2019)
def target_encode_kflod(df, cols, target_feature):
    for train_ind,val_ind in tqdm(kf.split(df)): # val_ind是K中的1块数据的索引，而train_ind是剩下的K-1块数据的索引 
        df_ = pd.DataFrame()
        df_['user_id']= df['user_id']
        for col in cols:
            # 用K-1块数据计算Target encoding，记录到字典
            grouped = df.iloc[train_ind][[col,target_feature]].groupby(col)[target_feature]
            # 用刚刚计算出的映射对这1块内容做Target encoding
            df_.iloc[val_ind,col+'_target_encoded_mean'] = df.iloc[val_ind][col].replace(dict(grouped.mean())).values
            df_.iloc[val_ind,col+'_target_encoded_median'] = df.iloc[val_ind][col].replace(dict(grouped.median())).values
            df_.iloc[val_ind,col+'_target_encoded_std'] = df.iloc[val_ind][col].replace(dict(grouped.std())).values
    return df_

# K折目标编码gender
train_kflod_target_gender = target_encode_kflod(target_input,feature,'gender')

# K折目标编码age
train_kflod_target_age = target_encode_kflod(target_input,feature,'age')

train_kflod_target_gender.to_csv("tc/train_kflod_target_gender.csv")
train_kflod_target_age.to_csv("tc/train_kflod_target_age.csv")