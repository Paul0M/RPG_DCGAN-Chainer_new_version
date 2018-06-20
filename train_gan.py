#!usr/bin/python

import os
import sys
import numpy as np
import cupy as cp
from RPGCharacters_util import RPGCharacters
from gan import Generator,Discriminator
import chainer
from chainer import Variable,cuda,optimizers,serializers
from PIL import Image
import random
random.seed(0)

save_path=sys.argv[1]
if not os.path.exists(save_path):
  os.mkdir(save_path)
if not os.path.exists(f"{save_path}/model"):
  os.mkdir(f"{save_path}/model")

def clip(a):
  return 0 if a<0 else (255 if a>255 else a)

def array_to_img(im):
  im = im*255
  im = np.vectorize(clip)(im).astype(np.uint8)
  im=im.transpose(1,2,0)
  img=Image.fromarray(im)
  return img

def save_img(img_array,save_path): #save from np.array (3,height,width)
  img = array_to_img(img_array)
  img.save(save_path)

Gen = Generator()
Dis = Discriminator()

gpu = 1 
if gpu>=0:
    xp = cuda.cupy
    cuda.get_device(gpu).use()
    Gen.to_gpu()
    Dis.to_gpu()
else:
    xp = np

optG = Gen.make_optimizer()
optD = Dis.make_optimizer()
optG.setup(Gen)
optD.setup(Dis)

real = RPGCharacters()
trainsize=real.train_size
testsize=real.test_size

batchsize = 64
max_epoch = 100
for epoch in range(max_epoch):
  loss_fake_gen = 0.0
  loss_fake_dis = 0.0
  loss_real_dis = 0.0
  n_fake_gen = 0
  n_fake_dis = 0
  n_real_dis = 0

  with chainer.using_config('train', True):
    for data,charaid,poseid in real.gen_train(batchsize):
      rand_ = random.uniform(0,1)
      B = data.shape[0]
      if rand_ < 0.2:
          Dis.cleargrads()

          x = Variable(xp.array(data))
          label_real = Variable(xp.ones((B,1),dtype=xp.int32))

          y, loss = Dis(x,label_real)
          loss_real_dis += loss.data
          loss.backward()
          optD.update()
          n_real_dis += B
      elif rand_ < 0.4:
          Dis.cleargrads()

          z = Gen.generate_hidden_variables(B)
          x = Gen(Variable(xp.array(z)))
          label_real = Variable(xp.zeros((B,1),dtype=xp.int32))
          y, loss = Dis(x,label_real)
          loss_fake_dis += loss.data
          loss.backward()
          optD.update()
          n_fake_dis += B
      else:
          Gen.cleargrads()
          Dis.cleargrads()

          z = Gen.generate_hidden_variables(B)
          x = Gen(Variable(xp.array(z)))
          label_fake = Variable(xp.ones((B,1),dtype=xp.int32))
          y, loss = Dis(x,label_fake)
          loss_fake_gen += loss.data
          loss.backward()
          optG.update()
          n_fake_gen += B
      sys.stdout.write(f"\rtrain... epoch{epoch}, {n_real_dis+n_fake_dis+n_fake_gen}/{trainsize}")
      sys.stdout.flush()
  
  
  with chainer.using_config('train', False), chainer.no_backprop_mode():
    z = Gen.generate_hidden_variables(batchsize)
    x = Gen(Variable(xp.array(z))) #(B,3,64,64) B:batchsize
    x.to_cpu()
    tmp = np.transpose(x.data,(1,0,2,3)) #(3,B,64,64)
    img_array=[]
    for i in range(3):
      img_array2=[]
      for j in range(0,batchsize,8):
        img=tmp[i][j:j+8]
        img=np.transpose(img.reshape(64*8,64),(1,0))
        img_array2.append(img)
      img_array2=np.array(img_array2).reshape(int(batchsize/8*64),8*64)
      img_array.append(np.transpose(img_array2,(1,0)))
    img_array = np.array(img_array)
    print("\nsave fig...")
    save_img(img_array,f"{save_path}/{str(epoch).zfill(3)}.png")  
    print(f"fake_gen_loss:{loss_fake_gen/n_fake_gen}(all/{n_fake_gen}), \
          fake_dis_loss:{loss_fake_dis/n_fake_dis}(all/{n_fake_dis}), \
          real_dis_loss:{loss_real_dis/n_real_dis}(all/{n_real_dis})") #losses are approximated values
    print('save model ...')
    prefix = f"{save_path}/model/str(epoch).zfill(3)"
    if os.path.exists(prefix)==False:
      os.mkdir(prefix)        
    serializers.save_npz(f"{prefix}/Geights", Gen.to_cpu()) 
    serializers.save_npz(f"{prefix}/Goptimizer", optG)
    serializers.save_npz(f"{prefix}/Dweights", Dis.to_cpu())
    serializers.save_npz(f"{prefix}/Doptimizer", optD)
    Gen.to_gpu()
    Dis.to_gpu()

  real_belief_mean = 0.0
  fake_belief_mean = 0.0
  for j,(data,charaid,poseid) in enumerate(real.gen_test(batchsize)):
        x = Variable(xp.array(data))  
        B = x.shape[0]      
        label = Variable(xp.ones((B,1),dtype=xp.int32)) 
        with chainer.using_config('train', False), chainer.no_backprop_mode():
          y, loss = Dis(x,label)
          real_belief_mean += xp.sum(y.data)
          sys.stdout.write(f"\rtest real...{j}/{testsize/batchsize}")
          sys.stdout.flush()
  print(f" test real belief mean:{real_belief_mean/testsize}({real_belief_mean}/{testsize})")
  for j,(data,charaid,poseid) in enumerate(real.gen_test(batchsize)):
        z = Gen.generate_hidden_variables(batchsize)
        x = Gen(Variable(xp.array(z)))
        label = Variable(xp.zeros((batchsize,1),dtype=xp.int32))
        with chainer.using_config('train', False), chainer.no_backprop_mode():      
          y, loss = Dis(x,label)
          fake_belief_mean += xp.sum(y.data) 
          sys.stdout.write(f"\rtest fake...{j}/{testsize/batchsize}")
          sys.stdout.flush()
  print(f" test fake belief mean:{fake_belief_mean/testsize}({fake_belief_mean}/{testsize})")
        

