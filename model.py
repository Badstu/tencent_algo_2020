import data
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class lgb_model():
    def __init__(self, model_kind="gender"):
        '''
        @train_dataset: lgb.Dataset(X, y)
        @valid_dataset: lgb.Dataset(X, y)
        '''
#         self.train_dataset = train_dataset
#         self.valid_dataset = valid_dataset
        
        self.params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
#             'max_depth': 7,
            'num_leaves': 80,  # 叶子节点数
            'learning_rate': 0.1,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.8,  # 建树的样本采样比例
            'lambda_l2': 0.01,
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
#         self.categorical_feature = [1, 3, 4, 5, 6, 7]
        self.num_interations = 200
        
        self.model_kind = model_kind
        if self.model_kind == "gender":
            self.params["metric"] = {"binary_logloss", "auc"}
            self.params["objective"] = "binary"
            self.model_save_path = "checkpoints/gender_model.pkl"
        elif self.model_kind == "age":
            self.params["metric"] = {"softmax"}
            self.params["objective"] = "multiclass"
            self.params['num_class'] = 10
            self.model_save_path = "checkpoints/age_model.pkl"
            self.num_interations = 1000
            self.num_leaves = 255

    def train(self, train_dataset, valid_dataset):
        self.gbm = lgb.train(self.params,
                        train_dataset,
                        num_boost_round=self.num_interations,
                        valid_sets=valid_dataset,
                        early_stopping_rounds=10)
#                         categorical_feature=self.categorical_feature)        
    
    def get_model(self):
        return self.gbm
    
    def save_model(self, path=None):
        if path == None:
            self.gbm.save_model(self.model_save_path)
        else:
            self.gbm.save_model(path)
        
    def load_model(self):
        self.gbm = lgb.Booster(model_file=self.model_save_path)
        
    def predict(self, input_features):
        pred = self.gbm.predict(input_features, num_iteration=self.gbm.best_iteration)
        return pred
    
    def transform_pred(self, pred):
        if self.model_kind == "gender":
            record_pred_label = pred.copy()
            record_pred_label[pred >= 0.5] = 1
            record_pred_label[pred < 0.5] = 0
            record_pred_label = record_pred_label.astype(int)
        elif self.model_kind == "age":
            record_pred_label = [list(x).index(max(x)) for x in pred]
            
        return record_pred_label
    
#     def measure(self, record_pred_label, data_record, data_user):
#         uni_pred, uni_acc = data.measure_unique_user(record_pred_label, data_record, data_user, self.model_kind)
#         return uni_pred, uni_acc

class lstm_model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(lstm_model, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.word_embeddings = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)
        self.dropout = nn.Dropout(p=0.6)
        

    def forward(self, x):
#         embeds = self.word_embeddings(x)
#         lstm_out, _ = self.lstm(embeds)
        
        h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        x_embed = self.word_embeddings(x)
        output, hidden_state = self.lstm(x_embed, (h0, c0))
        output = output.contiguous().view(output.shape[0] * output.shape[1], -1)
        output = self.fc(self.dropout(output))
        
        return output