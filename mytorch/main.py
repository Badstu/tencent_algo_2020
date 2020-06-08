import sys
sys.path.append('/home/sayhi/workspaces/tencent_algo_2020/')

import time
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

import torch
import torch.nn as nn
import torchvision
from torchnet import meter
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from mytorch.dataset import RecordDataset
from mytorch.lstm import ClickRNN

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def collate_fn(data):
    data.sort(key = lambda x: x[0], reverse=True)
    t = [x[0] for x in data]
    embed_features = [x[1] for x in data]
    age = [x[2] for x in data]
    gender = [x[3] for x in data]

    embed_features = rnn_utils.pad_sequence(embed_features, batch_first=True, padding_value=0)
    # embed_features = rnn_utils.pack_padded_sequence(embed_features, lengths=t, batch_first=True)
    return t, embed_features, age, gender

def transform_pred(pred, model_kind):
#     pred = pred.numpy()
    if model_kind == "gender":
        record_pred_label = pred.copy()
        record_pred_label[pred >= 0.5] = 1
        record_pred_label[pred < 0.5] = 0
        record_pred_label = record_pred_label.astype(int)
    elif model_kind == "age":
        record_pred_label = [list(x).index(max(x)) for x in pred]

    return record_pred_label

def main(train_list, valid_list, model_type="gender"):
    batch_size = 128
    hidden_dim = 200
    train_on_gpu=True
    device = torch.device("cuda") if train_on_gpu else torch.device("cpu")
    
    model_path = None
    lr = 1e-4
    lr_decay = 0.9
    weight_decay = 1e-5
    
    max_epoch = 50
    
    train_dataset = RecordDataset(train_list)
    valid_dataset = RecordDataset(valid_list)
#     test_dataset = RecordDataset(test_list)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=8, drop_last=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=8, drop_last=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True, 
#                              num_workers=8, drop_last=True, collate_fn=collate_fn)
    
    print(len(train_dataset), len(valid_dataset))
    
    if model_type == "gender":
        model = ClickRNN(input_dim=800, hidden_dim=hidden_dim, output_dim=1, 
                         n_layers=2, batch_size=batch_size, train_on_gpu=train_on_gpu)
    elif model_type == "age":
        model = ClickRNN(input_dim=800, hidden_dim=hidden_dim, output_dim=10, 
                         n_layers=2, batch_size=batch_size, train_on_gpu=train_on_gpu)
        

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.99)
    )
    
    if model_path != None:
        map_location = lambda storage, loc: storage
        checkpoint = torch.load(model_path, map_location=map_location)
        model.load_state_dict(checkpoint["model"])
        last_epoch = checkpoint["epoch"]
        lr = checkpoint["lr"]
        optimizer.load_state_dict(checkpoint["optimizer"])

    model = model.to(device)
    
    last_epoch = -1
    previous_loss = 1000
    best_test_acc = 0
    best_model = None
    
    for epoch in range(max_epoch):
        if epoch < last_epoch:
            continue

        # ========================================START TRAIN============================================
        # train
        model.train()
        if model_type == "gender":
            criterion = nn.BCEWithLogitsLoss()
        elif model_type == "age":
            criterion = nn.CrossEntropyLoss()

        loss_meter = meter.AverageValueMeter()
        acc_meter = meter.AverageValueMeter()
        loss_meter.reset()
        acc_meter.reset()

        for ii, (lengths, embed_features, age, gender) in tqdm(enumerate(train_loader)):
            if max(lengths) > 400:
                continue
            torch.cuda.empty_cache()
            # input: [batch, time_step, input_size]
            if model_type == "gender":
                label = torch.Tensor(gender) - 1
            elif model_type == "age":
                label = torch.Tensor(age) - 1
            embed_features = embed_features.to(device)
            label = label.long()
            label = label.to(device)

            out, _ = model(embed_features)

            loss = criterion(out, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            acc = accuracy_score(label.cpu().data, transform_pred(out.cpu().data, model_type))
            acc_meter.add(acc)
            loss_meter.add(loss.item())
            
            if ii % 200 == 0:
                print("epoch:{epoch}, \
                lr:{lr:.5f}, \
                train_loss:{loss:.5f}, \
                train_acc:{acc:.5f}".format(epoch = epoch, 
                                            lr = lr, 
                                            loss = loss_meter.value()[0], 
                                            acc = acc_meter.value()[0]))

        print("epoch:{epoch}, lr:{lr:.5f}, train_loss:{loss:.5f}, train_acc:{acc:.5f}".format(epoch = epoch,
                                                                                            lr = lr,
                                                                                            loss = loss_meter.value()[0],
                                                                                            acc = acc_meter.value()[0]))
        # ========================================STOP TRAIN============================================
        # ========================================START VALID============================================
        model.eval()
        if model_type == "gender":
            criterion = nn.BCEWithLogitsLoss()
        elif model_type == "age":
            criterion = nn.CrossEntropyLoss()

        val_loss_meter = meter.AverageValueMeter()
        val_acc_meter = meter.AverageValueMeter()
        val_loss_meter.reset()
        val_acc_meter.reset()

        for ii, (lengths, embed_features, age, gender) in tqdm(enumerate(valid_loader)):
            if max(lengths) > 400:
                continue
            torch.cuda.empty_cache()
            # input: [batch, time_step, input_size]
            if model_type == "gender":
                label = torch.Tensor(gender) - 1
            elif model_type == "age":
                label = torch.Tensor(age) - 1

            embed_features = embed_features.to(device)
            label = label.long()
            label = label.to(device)

            out, _ = model(embed_features)

            loss = criterion(out, label)

            acc = accuracy_score(label.cpu().data, transform_pred(out.cpu().data, model_type))
            val_acc_meter.add(acc)
            val_loss_meter.add(loss.item())
            
            if ii % 200 == 0:
                print("epoch:{epoch}, \
                        lr:{lr:.5f}, \
                        val_loss:{loss:.5f}, \
                        val_acc:{acc:.5f}".format(epoch = epoch, 
                                                  lr = lr, 
                                                  loss = val_loss_meter.value()[0], 
                                                  acc = val_acc_meter.value()[0]))

        print("epoch:{epoch}, lr:{lr:.5f}, val_loss:{loss:.5f}, val_acc:{acc:.5f}".format(epoch = epoch,
                                                                                            lr = lr,
                                                                                            loss = val_loss_meter.value()[0],
                                                                                            acc = val_acc_meter.value()[0]))
        # ========================================STOP VALID============================================


        best_test_acc = max(best_test_acc, val_acc_meter.value()[0])

        if best_test_acc > val_acc_meter.value()[0]:
            best_test_acc = val_acc_meter.value()[0]
            best_model = model
        print("best_test_auc(val) is: ", best_test_acc)

        current_loss = loss_meter.value()[0]    
        print("current_loss: ", current_loss)
        if (current_loss > previous_loss) or ((epoch + 1) % 5) == 0:
            lr = lr * lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = current_loss

        # TODO 每save_every个epoch结束后保存模型参数+optimizer参数
        if (epoch + 1) % 10 == 0:
            prefix = "checkpoints/LSTM_epoch{}_".format(epoch+1)
            file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
            checkpoint = {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "model": model.state_dict(),
                "lr": lr
            }
            torch.save(checkpoint, file_name)

    # TODO 结束的时候保存final模型参数
    file_name = time.strftime('checkpoints/LSTM_final_%m%d_%H_%M_%S.pth')
    checkpoint = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict(),
        "lr": lr
    }
    torch.save(checkpoint, file_name)
    return best_model



if __name__ == "__main__":
    train_ad, train_click, train_user, test_ad, test_click = data.load_data()
    # train_record
    train_record = pd.merge(train_click, train_ad, on="creative_id")
    # test_record
    test_record = pd.merge(test_click, test_ad, on="creative_id")
    
    train_grouped = train_record.groupby("user_id")
    test_grouped = test_record.groupby("user_id")
    
    train_list = train_grouped.agg(list).reset_index()
    train_list = pd.merge(train_list, train_user, on="user_id")
    
    train_list, valid_list = train_test_split(train_list, test_size=0.33, random_state=42)
    
    print("start train!!!")
    best_age_model = main(train_list, valid_list, model_type="age")
    file_name = time.strftime('checkpoints/LSTM_best_age_%m%d_%H_%M_%S.pth')
    checkpoint = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "model": best_age_model.state_dict(),
        "lr": lr
    }
    torch.save(checkpoint, file_name)
    
    best_gender_model = main(train_list, valid_list, model_type="gender")
    file_name = time.strftime('checkpoints/LSTM_best_gender_%m%d_%H_%M_%S.pth')
    checkpoint = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "model": best_gender_model.state_dict(),
        "lr": lr
    }
    torch.save(checkpoint, file_name)