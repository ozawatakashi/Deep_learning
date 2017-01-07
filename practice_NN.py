#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 21:32:13 2016

@author: ozawa
"""

import numpy as np

import chainer

from chainer import cuda , Function , gradient_check ,  Variable , optimizers , serializers , utils 
from chainer import Link , Chain , ChainList
import chainer.functions as F 
import chainer.links as L
#%%
x1 = Variable(np.array([1] , dtype=np.float32))
x2 = Variable(np.array([2] , dtype=np.float32))
x3 = Variable(np.array([3] , dtype=np.float32))

x1 = Variable(np.array([1]).astype(np.float32))
x2 = Variable(np.array([2]).astype(np.float32))
x3 = Variable(np.array([3]).astype(np.float32))
#%%
z = (x1 - 2 * x2 - 1)**2 + (x2 * x3 - 1)**2 + 1
z.data

z.backward()

x1.grad
x2.grad
x3.grad
#%%
x = Variable(np.array([-1],dtype=np.float32))
F.sin(x).data
F.sigmoid(x).data
#%%
x=Variable(np.array([-0.5],dtype=np.float32))
z=F.cos(x)
z.data
z.backward()
x.grad
((-1)*F.sin(x)).data#確認(cos(x))'=-sin(x)
#%%


x=Variable(np.array([-1,0,1],dtype=np.float32)) 
z=F.sin(x) 
z.grad=np.ones(3,dtype=np.float32) 
z.backward()
x.grad
#%%
h = L.Linear(3,4)
h.W.data
h.b.data
x = Variable(np.array(range(6)).astype(np.float32).reshape(2,3))
x.data
y=h(x)
y.data
w=h.W.data
x0=x.data
x0.dot(w.T) + h.b.data # Y = h(x) の検算
#%%
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
l1 = L.Linear(4,3),
l2 = L.Linear(3,3),
)
        def __call__(self,x,y):
            fv = self.fwd(x,y)
            loss = F.mean_squared_error(fv,y) 
            return loss
        def fwd(self,x,y):
            return F.sigmoid(self.l1(x))
model=MyChain()#モデルを生成 
optimizer = optimizers.SGD() #最適化のアルゴリズムの選択
optimizer.setup(model) #アルゴリズムにモデルをセット
model.zerograds()#勾配の初期化
 loss = model(x,y)#順方向に計算して誤差を算出 
 loss.backward()#逆方向の計算、勾配の計算 
 optimizer.update()#パラメータを更新