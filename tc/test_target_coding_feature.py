import pandas as pd
import numpy as np
from tqdm import tqdm
# 数据读取
# 训练集
data_path = 'dataset/'
train_ad = pd.read_csv(data_path+ "train/ad.csv", na_values="\\N")
train_click = pd.read_csv(data_path+ "train/click_log.csv", na_values="\\N")
train_user = pd.read_csv(data_path+ "train/user.csv", na_values="\\N")
train_data = train_click.merge(train_ad, how="left", on="creative_id", )
train_data= train_data.merge(train_user, how="left", on="user_id", )
train_data.fillna(0,inplace=True)
train_data[['creative_id','ad_id','product_id','advertiser_id','industry']] = train_data[['creative_id','ad_id','product_id','advertiser_id','industry']].astype('object')
# 测试集
test_ad = pd.read_csv(data_path+"test/ad.csv", na_values="\\N")
test_click = pd.read_csv(data_path+"test/click_log.csv", na_values="\\N")
test_data = test_click.merge(test_ad, how="left", on="creative_id", )


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

# gender属性的目标编码
train_target_gender= target_encode(target_input,feature,'gender')

# age属性的目标编码
train_target_age = target_encode(target_input,feature,'age')

# test target coding
# df1：链接后的新表， df2:测试集表  col:属性列
def test_target_coding(df1,df2,cols):
    for col in tqdm(cols):
        count_data = df1[[col,col+'_target_encoded_mean',col+'_target_encoded_median',col+'_target_encoded_std']].drop_duplicates(keep='first')
        df2 = df2.merge(count_data,on=col)
    return df2

##### gender target表####
# 跟原表链接一下 gender特征列构建
new_train_data_gender = pd.concat([train_data, train_target_gender],axis =1)
new_test_data_gender = test_target_coding(new_train_data_gender,test_data,feature)
del new_train_data_gender
#  需要删除的列
drop_list = ['time','creative_id', 'click_times', 'ad_id', 'product_id',
       'product_category', 'advertiser_id', 'industry']
# 删除无关列
new_test_data_gender.drop(drop_list, axis=1, inplace=True)
# gender target encoding 特征表输出
new_test_data_gender =  new_test_data_gender.groupby('user_id').agg('median')
# new_test_data_gender['user_id'] = new_test_data_gender.index
new_test_data_gender.to_csv('test_target_gender.csv')


##### gender age表####
# 跟原表链接一下 age特征列构建
new_train_data_age = pd.concat([train_data, train_target_age],axis =1)
new_test_data_age = test_target_coding(new_train_data_age,test_data,feature)
del new_train_data_age
# 删除无关列
new_test_data_age.drop(drop_list, axis=1, inplace=True)
# age target encoding 特征表输出
new_test_data_age =  new_test_data_age.groupby('user_id').agg('median')
# new_test_data_age['user_id'] = new_test_data_age.index
new_test_data_age.to_csv('test_target_age.csv')