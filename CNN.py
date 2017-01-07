#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 15:39:52 2016

@author: ozawa
"""
# 必要なものをimport

import glob
import os
from PIL import Image
import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers ,Variable
import time

gpu_flag = 0
 
batchsize = 1
n_epoch = 10
# 画像の読み込み
print(os.getcwd()) #directryの確認

os.chdir('/Users/ozawa/Desktop/python/Deep_learning/img2')
imgname = glob.glob('*.jpg')


imglen = len(imgname)

arr = []
for i in range(imglen):
    arr.append(np.asarray(Image.open(imgname[i])))

# 画像を (nsample, channel, height, width) の4次元テンソルに変換
arr = np.asarray(arr).astype(np.float32)
arr = arr.transpose(0,3,1,2)

#%%
X = arr
#答えデータの生成
makelen = len(glob.glob('-1*.jpg'))
staylen = len(glob.glob('0*.jpg'))
katilen = len(glob.glob('1*.jpg'))

make = np.zeros(makelen) -1
stay = np.zeros(staylen)
kati = np.zeros(katilen) +1

y = np.asarray([make,stay,kati])
y = y.reshape(imglen).astype(np.int32)


# ピクセルの値を0.0-1.0に正規化
X /= X.max()
 
#%%
# 訓練データとテストデータに分割
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
# X_test, X_train = np.split(X,[imglen / 3])
# y_test, y_train = np.split(y,[imglen / 3])

N = y_train.size
N_test = y_test.size

#%%
# 図示化
import pylab
import matplotlib.pyplot as plt
 
plt.imshow(X_train[11][1], cmap=pylab.cm.gray_r, interpolation='nearest')
#%%
XX = Variable(X)
YY = XX.data
YY2 = F.max_pooling_2d(YY2,2)
YY1 = YY2.data

# modelの定義


 
model = chainer.FunctionSet(conv1=F.Convolution2D(3, 3, 51,stride = 1),   # 入力1枚、出力20枚、フィルタサイズ5ピクセル
                            conv2=F.Convolution2D(3, 3,16,stride = 1),  # 入力20枚、出力50枚、フィルタサイズ5ピクセル,
                            l1=F.Linear(75, 3),             # 入力800ユニット、出力500ユニット
                            l2=F.Linear(3, 3))              # 入力500ユニット、出力10ユニット

def forward(x_data, y_data, train=True):
    x,t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
    h1 = F.max_pooling_2d(F.relu(model.conv2(h)), 2)
    h2 = F.dropout(F.relu(model.l1(h1)))
    y = model.l2(h2)


    if train:
        return F.softmax_cross_entropy(y, t)
    else:
        return F.accuracy(y, t)
 
optimizer = optimizers.Adam()
optimizer.setup(model)

fp1 = open("accuracy.txt", "w")
fp2 = open("loss.txt", "w")
 
fp1.write("epoch\ttest_accuracy\n")
fp2.write("epoch\ttrain_loss\n")

#%%/
# 学習の実行
start_time = time.clock()
for epoch in range(1, n_epoch + 1):
    print ("epoch: %d" % epoch)
 
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = np.asarray(X_train[perm[i:i + batchsize]])
        y_batch = np.asarray(y_train[perm[i:i + batchsize]])
 
        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(y_batch)
 
    print ("train mean loss: %f" % (sum_loss / N))
    fp2.write("%d\t%f\n" % (epoch, sum_loss / N))
    fp2.flush()
 
    sum_accuracy = 0
    for i in range(0, N_test, batchsize):
        x_batch = np.asarray(X_test[i:i + batchsize])
        y_batch = np.asarray(y_test[i:i + batchsize])
 
        acc = forward(x_batch, y_batch, train=False)
        sum_accuracy += float(acc.data) * len(y_batch)
 
    print ("test accuracy: %f" % (sum_accuracy / N_test))
    fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
    fp1.flush()
    
end_time = time.clock()
print( end_time - start_time)
#%%
# traindetaの学習成果

forward(X_train,y_train,train = False).data
 #%%
 # ファイルを閉じる
fp1.close()
fp2.close()
#%%
"""
import cPickle
# CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
model.to_cpu()
 cPickle.dump(model, open("model.pkl", "wb"), -1)

"""