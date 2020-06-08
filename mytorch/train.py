from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from torchnet import meter


def train(model, data_loader, epoch, lr, optimizer, model_type="gender"):
    model.train()
    
    if model_type == "gender":
        criterion = nn.BCEWithLogitsLoss()
    elif model_type == "age":
        criterion = nn.CrossEntropyLoss()
    
    loss_meter = meter.AverageValueMeter()
    acc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    train_loss_list = []
    for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(data_loader)):
        # input: [batch, time_step, input_size](after embedding)
        max_seq_len = batch_seq.shape[1]
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)

        output, hidden_state = model(batch_seq) # [batch_size*max_seq_len, 110]

        if opt.model_name == "CNN_3D_mask":
            # output [batch_size*200, 1]
            # TODO mask output to predict
            next_question_number = batch_label[:, :, 0].view(-1).long()
            next_question_label = batch_label[:, :, 1].view(-1)

            label = []
            mask = torch.zeros_like(output)
            for i in range(opt.batch_size):
                start = i * max_seq_len
                len = batch_len[i]
                mask[start: start+len] = True
                label.extend(next_question_label[start:start+len])

            predict = torch.masked_select(output, mask.bool())
            label = torch.Tensor(label).to(opt.device)
            if label.sum() == label.shape[0] or label.sum() == 0:
                continue
            ##############
        else:
            # TODO mask output to predict
            next_question_number = batch_label[:, :, 0].view(-1).long()
            next_question_label = batch_label[:, :, 1].view(-1)

            mask = torch.zeros_like(output)
            label = []
            for i in range(opt.batch_size):
                start = i * max_seq_len
                len = batch_len[i]
                mask[range(start, start+len), next_question_number[start:start+len] - 1] = True
                label.extend(next_question_label[start:start+len])

            predict = torch.masked_select(output, mask.bool())
            label = torch.Tensor(label).to(opt.device)
            ##############

        loss = criterion(predict, label)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        auc = roc_auc_score(label.cpu().data, predict.cpu().data)
        auc_meter.add(auc)
        loss_meter.add(loss.item())

        train_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值
        if opt.vis and (ii) % opt.plot_every_iter == 0:
            vis.plot("train_loss", loss_meter.value()[0])
            vis.plot("train_auc", auc_meter.value()[0])
            vis.log("epoch:{epoch}, lr:{lr:.5f}, train_loss:{loss:.5f}, train_auc:{auc:.5f}".format(epoch = epoch,
                                                                                                lr = lr,
                                                                                                loss = loss_meter.value()[0],
                                                                                                auc = auc_meter.value()[0]))
    return loss_meter, auc_meter, train_loss_list

@torch.no_grad()
def valid(opt, vis, model, valid_loader, epoch):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    val_loss_list = []
    for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(valid_loader)):
        max_seq_len = batch_seq.shape[1]
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)

        output, hidden_state = model(batch_seq)

        if opt.model_name == "CNN_3D_mask":
            # output [batch_size*200, 1]
            # TODO mask output to predict
            next_question_number = batch_label[:, :, 0].view(-1).long()
            next_question_label = batch_label[:, :, 1].view(-1)

            label = []
            mask = torch.zeros_like(output)
            for i in range(opt.batch_size):
                start = i * max_seq_len
                len = batch_len[i]
                mask[start: start+len] = True
                label.extend(next_question_label[start:start+len])

            predict = torch.masked_select(output, mask.bool())
            label = torch.Tensor(label).to(opt.device)
            if label.sum() == label.shape[0] or label.sum() == 0:
                continue
            ##############
        else:
            # TODO mask output to predict
            next_question_number = batch_label[:, :, 0].view(-1).long()
            next_question_label = batch_label[:, :, 1].view(-1)

            mask = torch.zeros_like(output)
            label = []
            for i in range(opt.batch_size):
                start = i * max_seq_len
                len = batch_len[i]
                mask[range(start, start+len), next_question_number[start:start+len] - 1] = True
                label.extend(next_question_label[start:start+len])

            predict = torch.masked_select(output, mask.bool())
            label = torch.Tensor(label).to(opt.device)
            ##############

        loss = criterion(predict, label)
        auc = roc_auc_score(label.cpu().data, predict.cpu().data)

        auc_meter.add(auc)
        loss_meter.add(loss.item())

        val_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值

        if opt.vis:
            vis.plot("valid_loss", loss_meter.value()[0])
            vis.plot("valid_auc", auc_meter.value()[0])

    if opt.vis:
        vis.log("epoch:{epoch}, valid_loss:{loss:.5f}, valid_auc:{auc:.5f}".format(epoch=epoch,
                                                                            loss=loss_meter.value()[0],
                                                                            auc=auc_meter.value()[0]))
    return loss_meter, auc_meter, val_loss_list
