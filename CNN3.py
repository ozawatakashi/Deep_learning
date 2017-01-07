#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 00:08:43 2016

@author: ozawa
"""
import os
import glob
from PIL import Image
import numpy as np
from chainer import optimizers
from chainer import  Chain
from chainer import iterators, serializers
from chainer import report, training , datasets
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
import time
from chainer.training import extensions
from sklearn.cross_validation import train_test_split
#%%
# titanicdataの読み込み
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
# X_test, X_train = np.split(X,[imglen / 3])
# y_test, y_train = np.split(y,[imglen / 3])

N = y_train.size
N_test = y_test.size

 

#%%
train = tuple_dataset.TupleDataset(X_train, y_train)
test = tuple_dataset.TupleDataset(X_test, y_test)
#%%
train_iter = iterators.SerialIterator(train, 1)
test_iter = iterators.SerialIterator(test, 1,repeat=False, shuffle=False)

#%%
class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
                            conv1=F.Convolution2D(3, 3, 51,stride = 1),   # 入力1枚、出力20枚、フィルタサイズ5ピクセル
                            conv2=F.Convolution2D(3, 3,16,stride = 1),  # 入力20枚、出力50枚、フィルタサイズ5ピクセル,
                            l1=F.Linear(75, 3),             # 入力800ユニット、出力500ユニット
                            l2=F.Linear(3, 3)
        )
    def __call__(self, x, train=True):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h1 = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h2 = F.dropout(F.relu(self.l1(h1)))
        y = self.l2(h2)
        return F.sigmoid(y)
#%%             
model = L.Classifier(MLP())
optimizer = optimizers.Adam()
optimizer.setup(model)
#%%
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (15, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport(trigger = (1,'epoch')))
trainer.extend(extensions.PrintReport( ['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
#%%
