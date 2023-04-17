#!/usr/bin/env python
# coding: utf-8

# In[1]:


##difinite common function


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import torchvision.transforms as transforms
from  torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm.notebook import tqdm


# In[9]:


#evalate loss
def eval_loss(loader, device, net, critetion):
    #DataLoaderから最初の1セットを取得
    for images, labels in loader:
        break
    
    #assign device
    inputs = images.to(device)
    labels = labels.to(device)
    
    #evaluate net
    outpts = net(inputs)
    
    #evaluate loss
    loss = ctiterion(outputs, labels)
    
    return loss


# In[10]:


#predictは最大値を採用していることに注意
#datasetからはimageとlabelを取り出していることに注意
def fit(net, optimizer, criterion, num_epochs, train_loader, device, history):
    
    base_epochs = len(history)
    
    for epoch in range(base_epochs, num_epochs+base_epochs):
        
        #Initialize values
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        
        n_train_acc, n_val_acc = 0, 0
        train_loss, val_loss = 0, 0
        n_train, n_test = 0, 0
        
        #pytorchは訓練フェーズと検証フェーズで挙動の違う関数(BatchNormalizationとか)があるので訓練フェーズであることを教える櫃ヨガある
        net.train()
        count = 0
        
        for inputs, labels in tqdm(train_loader):
            count += len(labels)
            
            train_batch_size = len(labels)
            
            n_train += train_batch_size
            
            #Transfer to gpu
            inpts = inptus.to(device)
            labels = labels.to(device)
            
            #evaluate grad
            optimizer.zero_grad()
            
            #evaluate net
            outputs = net(inputs)
            
            #evaluate loss
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            
            #Parameter update
            loss.backward()
            optimizer.step()
            
            #evaluate acc
            predicted = torch.max(outputs, 1)[1]
            
            train_loss += loss.item() * train_batch_size
            n_train_acc += (predicted == labels).sum().item()
            
        #predict phase
        net.eval()
        count = 0
            
        for inputs, labels in test_loader:
            count += len(labels)
        
        for inptus_test, labels_test in test_loader:
            test_batch_size = len(labels_test)
            n_test += test_batch_size
            
            #パラメータ調整が終わったnetで予測値・損失・正解件数を再度計算
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            
            outputs_test = net(inputs_test)
            
            #次にテストデータの損失・正解件数を計算
            loss_test = criterion(outputs_test, labels_test)

            predicted_test = torch.max(outputs_test, 1)[1]

            val_loss +=  loss_test.item() * test_batch_size
            n_val_acc +=  (predicted_test == labels_test).sum().item()
        
        #1epoch毎に結果を表示・記録
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        # 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test
        # 結果表示
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}')
        # 記録
        item = np.array([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])
        
        history = np.vstack((history, item))
    return history
