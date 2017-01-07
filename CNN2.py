#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:29:24 2016

@author: ozawa
"""

# 必要なものをimport
import argparse
import numpy as np
import six
import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers ,Variable

import data
from PIL import Image
import time



parser = argparse.ArgumentParser(description='Chainer example: HISTORY')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()


batchsize = 1
n_epoch = 1
# 画像の読み込み
img = np.array( Image.open('/home/ec2-user/02010#12#02#04#00.jpg') )
img2 = np.array( Image.open('/home/ec2-user/02010#12#06#08#00.jpg') )
img3 = np.array( Image.open('/home/ec2-user/02010#12#06#16#00.jpg') )
img4 = np.array( Image.open('/home/ec2-user/02010#12#13#20#00.jpg') )
img5 = np.array( Image.open('/home/ec2-user/02010#12#21#00#00.jpg') )
img = img.transpose(2, 0, 1)
img2 = img2.transpose(2, 0, 1)
img3 = img3.transpose(2, 0, 1)
img4 = img4.transpose(2, 0, 1)
img5 = img5.transpose(2, 0, 1)
arr = np.asarray([img,img2,img3,img4,img5])

#arr = np.append(arr, img)
#arr = np.append(arr, img2)
#%%
# plt.imshow(X_train[2][1], cmap=pylab.cm.gray_r, interpolation='nearest')
#%%
X = arr
#答えデータの生成
y1 = np.zeros(5)
y2 = np.ones(5)
y3 = np.zeros(5)
y = np.asarray([y1,y2,y3])
y = y.transpose(1,0)

X = X.astype(np.float32)
y = y.astype(np.int32)
 
# ピクセルの値を0.0-1.0に正規化
X /= X.max()
 
# 訓練データとテストデータに分割
X_train, X_test = np.split(X,[4])
y_train , y_test = np.split(y,[4])
N = y_train.size
N_test = y_test.size
 
# 画像を (nsample, channel, height, width) の4次元テンソルに変換
# MNISTはチャンネル数が1なのでreshapeだけでOK


model = chainer.FunctionSet(conv1=F.Convolution2D(3, 20, (101,150)),   # 入力1枚、出力20枚、フィルタサイズ5ピクセル
                            conv2=F.Convolution2D(20, 50, (101,150)),  # 入力20枚、出力50枚、フィルタサイズ5ピクセル
                            conv3=F.Convolution2D(50, 20, (21,30)),
                            conv4=F.Convolution2D(20, 20, (5,8)),
                            l1=F.Linear(1100, 500),             # 入力800ユニット、出力500ユニット
                            l2=F.Linear(500, 3))              # 入力500ユニット、出力10ユニット

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()



def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
    h = F.max_pooling_2d(F.relu(model.conv2(h)), 2)
    h = F.max_pooling_2d(F.relu(model.conv3(h)), 2)
    h = F.max_pooling_2d(F.relu(model.conv4(h)), 2)
    h = F.dropout(F.relu(model.l1(h)), train=train)
    y = model.l2(h)
    if train:
        return F.softmax_cross_entropy(y, t)
    else:
        return F.accuracy(y, t)
 
optimizer = optimizers.Adam()
optimizer.setup(model)

#%%
# 訓練ループ
start_time = time.clock()
#for epoch in range(1, n_epoch + 1):
#    print ("epoch: %d" % epoch)
 
#    perm = np.random.permutation(N)
for i in range(1):
        x = np.asarray(X_train)
        yv = np.asarray(y_train)
        optimizer.zero_grads()
        loss = forward(x, yv)
        loss.backward()
        optimizer.update()
xt = Variable(X_test, volatile='on')
yy = model.fwd(xt)

ans = yy.data
nrow, ncol = ans.shape
ok = 0
for i in range(nrow):
    cls = np.argmax(ans[i,:])
    print (ans[i,:], cls)            
    if cls == y_test[i]:
        ok += 1
        
print (ok, "/", nrow, " = ", (ok * 1.0)/nrow)        
    
end_time = time.clock()
print( end_time - start_time)
 
#%%
"""
import cPickle
# CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
model.to_cpu()
 cPickle.dump(model, open("model.pkl", "wb"), -1)

"""