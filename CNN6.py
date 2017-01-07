import os
from PIL import Image
import numpy as np
import chainer
import chainer.functions as F
from chainer import  Chain
from chainer import cuda
from chainer import datasets
from chainer.datasets import tuple_dataset
import chainer.links as L
from chainer import optimizers
from chainer.training import extensions
from chainer import iterators
from chainer import training

#import pandas as pd

os.environ["CHAINER_TYPE_CHECK"] = "0" #type_checkしない

#======================CONST===========================
#======================================================
gpu_flag = 0
batchsize = 1
n_epoch = 5
#======================================================

print(os.getcwd())

os.chdir('/home/ec2-user/image8bitbkup')
imgname = glob.glob('*.png')
#imglen = len(imgname)
imglen = 900

arr = []
anserArray = []
for i in range(imglen):
   print (imgname[i])
   filenamestring=imgname[i]
   ans = filenamestring[0:-20]
   anserArray.append(ans)
   print(ans)
   imagedata=Image.open(imgname[i])
   imgarr=[np.asarray(imagedata)]
   arr.append(np.asarray(imgarr))

arr = np.asarray(arr)

pictureArray = arr.reshape(imglen, 1, 795, 492).astype(np.float32)

pictureArray /= pictureArray.max()



npAnserArray=np.asarray(anserArray)
npAnserArray = npAnserArray.reshape(imglen).astype(np.int32)


#%%
train,test= datasets.split_dataset_random(chainer.datasets.TupleDataset(pictureArray,npAnserArray),700)
#%%
train_iter = iterators.SerialIterator(train, 50)
test_iter = iterators.SerialIterator(test, 1,repeat=False, shuffle=False)
#%%
class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
                            conv1=F.Convolution2D(1, 30, 5,stride = 1),   # 入力3 枚、出力30枚、フィルタサイズ5ピクセル
                            conv2=F.Convolution2D(30, 30,6,stride = 1),# 入力30>枚、出力30枚、フィルタサイズ5ピクセル
                            conv3=F.Convolution2D(30, 30,5,stride = 1),# 入力30>枚、出力30枚、フィルタサイズ6ピクセル
                            conv4=F.Convolution2D(30, 30,5,stride = 1),# 入力30>枚、出力30枚、フィルタサイズ5ピクセル
                            conv5=F.Convolution2D(30, 30,6,stride = 1),# 入力30>枚、出力30枚、フィルタサイズ6ピクセル
                            l1=F.Linear(None, 500),# 入力960ユニット、出力500ユニット
                            l2=F.Linear(500, 20),               # 入力500ユニット、出力3ユニット
                            l3=F.Linear(20,3)
        )
    def __call__(self, x, train=True):
        h = F.max_pooling_2d(x, 2)
        h1 = F.max_pooling_2d(F.relu(self.conv1(h)), 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2)
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), 2)
        h4 = F.max_pooling_2d(F.relu(self.conv4(h3)), 2)
        h5 = F.max_pooling_2d(F.relu(self.conv5(h4)), 2)
        y = F.relu(self.l1(h5))
        y1 = F.relu(self.l2(y))
        return F.sigmoid(self.l3(y1))
#%%         
model = L.Classifier(MLP())
chainer.cuda.get_device(0).use()
model.to_gpu()
    # Setup an optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)
#%%
updater = training.StandardUpdater(train_iter, optimizer, device= gpu_flag)
trainer = training.Trainer(updater, (10, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_flag))
trainer.extend(extensions.LogReport(trigger = (1,'epoch')))
trainer.extend(extensions.PrintReport( ['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()