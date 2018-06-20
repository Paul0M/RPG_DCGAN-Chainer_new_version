# RPG_DCGAN-Chainer_new_version
Simple DCGAN RPG Characters with Chainer up-to-date(now 4.1.0)

Credit to [[SeitaroShinagawa](https://github.com/SeitaroShinagawa/DCGAN-chainer)]


# DCGAN-chainer
DCGAN simple implementation using chainer [[Paper](https://arxiv.org/abs/1511.06434)]  

## Contents  
train_gan.py: main code  
gan.py: network definition (quoted from [[Chainerを使ってコンピュータにイラストを描かせる](http://qiita.com/rezoolab/items/5cc96b6d31153e0c86bc)]) (in Generator, tanh->sigmoid)  
RPGCharacters_util.py: Utility of dataset([[Yurudorashiru free image resource](http://yurudora.com/tkool/)])  

## Requirements (via pip install):  
Chainer [[[link](http://chainer.org/)]] verified this code works in version 4.1.0  
pillow  
numpy  
scipy    

## How to run   
First, download dataset from "戦闘ユニット素材　ダウンロード(181.0MB)" in [[http://yurudora.com/tkool/](http://yurudora.com/tkool/)]  
(This dataset has about 62,000 images and each image is 64x64. It is same image size to the paper. So, this dataset is desirable to try simple GAN at first)    
After you get 3_sv_actors_20160915 directory, put it to the same place to this code.  
Fill out "image_root" path in train_gan.py (You can see from L.44 as below).  
```python  
image_root="/path/to/3_sv_actors_20160915" #need to be modified 
img_list=[]  
with open(image_root+"/list.txt",'r') as f:  
  for line in f:  
    img_list.append(line.strip())  
```  
As you can see, you need to create image list "list.txt" as follows,  
```   
cd /path/to/3_sv_actors_20160915  
ls 3_sv_actors >> list.txt  
``` 
Run the code.
```    
python train_gan.py /path/to/save  
```  
(`/path/to/save` means where you want to save the model and generated images every epoch)  

## Random generated images  
**epoch0**

![epoch0](https://github.com/Paul0M/RPG_DCGAN-Chainer_new_version/blob/master/images/000.png)

**epoch1**

![epoch1](https://github.com/Paul0M/RPG_DCGAN-Chainer_new_version/blob/master/images/001.png)

**epoch100**

![epoch100](https://github.com/Paul0M/RPG_DCGAN-Chainer_new_version/blob/master/images/100.png)

**epoch101**

![epoch101](https://github.com/Paul0M/RPG_DCGAN-Chainer_new_version/blob/master/images/101.png)

**epoch198**

![epoch198](https://github.com/Paul0M/RPG_DCGAN-Chainer_new_version/blob/master/images/198.png)

**epoch199**

![epoch199](https://github.com/Paul0M/RPG_DCGAN-Chainer_new_version/blob/master/images/199.png)

