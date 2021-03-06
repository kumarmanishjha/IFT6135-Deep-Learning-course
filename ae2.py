# -*- coding: utf-8 -*-
"""AE2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WeR6gsv4Y1LxDREPbWsMk-ah-kfkb968
"""

# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')

import torch
from torch import nn
from skimage import io, transform
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision

device = torch.device('cuda')

trans=torchvision.transforms.Compose([torchvision.transforms.Resize((64,64)), torchvision.transforms.ToTensor()])

ds=torchvision.datasets.STL10('drive/My Drive', split='train', transform=trans, target_transform=None, download=True)

"""# Olivetti

##Download olivetti (one time)
"""

!wget https://cs.nyu.edu/~roweis/data/olivettifaces.gif
!mv olivettifaces.gif 'drive/My Drive'

"""##Make DS, loader, ..."""

class Olive(Dataset):
  def __len__(self):
    return 400
  def __init__(self, transform=None):
    im=io.imread('drive/My Drive/olivettifaces.gif')
    im=np.array(im)
    H,W=im.shape
    im2=np.zeros((H,W,3),'uint8')
    im2[:,:,0]=im
    im2[:,:,1]=im
    im2[:,:,2]=im
    self.im=im
    self.im2=im2
    self.transform=transform
    
  def __getitem__(self, index):
    i=index//20
    j=index%20
    #i,j=19,19
    subim=self.im2[57*i:57*(i+1),47*j:47*(j+1)]
    subim=torchvision.transforms.ToPILImage()(subim)
    return self.transform(subim), 0 #0 is dummy value for target

"""#load data"""

trans=torchvision.transforms.Compose([torchvision.transforms.Resize((64,64)),
                                      torchvision.transforms.ToTensor()])
oliveds=Olive(transform=trans)

#sampler=torch.utils.data.RandomSampler(oliveds,True,300)

trainloader=DataLoader(oliveds,16,True)

loader=DataLoader(oliveds,16,True,pin_memory=True)

#Functions to convert between tensor/numpy
def batchTensor(im):
  if im.dtype==np.uint8:
    im=np.array(im,np.float32)
    im=im/255
  im=torch.tensor(im, dtype=torch.float32,device=device)
  im=im.unsqueeze(0)
  im.transpose_(1,3)
  return im

def batchTensorToNP(batch):
  batch=batch[0]
  batch.transpose_(0,2)
  im=batch.detach().cpu().numpy()
  if im.dtype!=np.uint8:
    im=255*im
    im=np.array(im,np.uint8)
  return im

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv=nn.Conv2d(3,1,5)
        self.conv2=nn.Conv2d(1,1,5)
        self.deconv2=nn.ConvTranspose2d(1,1,5)
        self.deconv=nn.ConvTranspose2d(1,3,5)
        
    def forward(self, input):
        output=self.conv(input)
        output=self.conv2(output)
        output=self.deconv2(output)
        output=self.deconv(output)
        return output
'''

def compLinSize(n):
  if n==96:
    return 24
  elif n==64:
    return 16
  else:
    raise(ValueError())

class Net(nn.Module):
  def __init__(self,n):
    super(Net, self).__init__()
    self.encoder=[
    nn.Conv2d(3,12,3),
    nn.ReLU(),
        
    nn.MaxPool2d(2,return_indices=True,padding=(1,1)),
        
    nn.Conv2d(12,12,3,padding=True),
    nn.ReLU(),
    
    nn.Conv2d(12,24,3,padding=True),
    nn.ReLU(),
    
    nn.MaxPool2d(2,return_indices=True),
        
    nn.Linear(24*compLinSize(n)**2,1568),
    #nn.Linear(2000,100)
    ]
    
    self.decoder=[
    #nn.Linear(100,2000),
    nn.Linear(1568,24*compLinSize(n)**2),    
        
    nn.MaxUnpool2d(2),
        
    nn.ConvTranspose2d(24,12,3,padding=True),
    nn.ReLU(),
        
    nn.ConvTranspose2d(12,12,3,padding=True),
    nn.ReLU(),    
        
    nn.MaxUnpool2d(2),
        
    nn.ConvTranspose2d(12,3,3,padding=True),     
    nn.ReLU()
    ]
    
    self.encoder=nn.ModuleList(self.encoder)
    self.decoder=nn.ModuleList(self.decoder)
    self.indStack=[]
    self.linStack=[]
    
    self.init_weights()
    
  def init_weights(self):
    for layer in self.encoder:
      if type(layer) is nn.Conv2d:
        nn.init.xavier_normal_(layer.weight)
        
    for layer in self.decoder:
      if type(layer) is nn.Conv2d:
        nn.init.xavier_normal_(layer.weight)
        
  def encode(self,inp):
    for layer in self.encoder:
      if type(layer) is nn.Linear:
        self.linStack.append(inp.shape)
        inp=inp.view(inp.shape[0],-1)
        
      if type(layer) is nn.MaxPool2d:
        inp, ind=layer(inp)
        self.indStack.append(ind)
      else:
        inp=layer(inp)
        
      #print(inp.shape)
    return inp
      
  def decode(self,inp):
    for layer in self.decoder:

      if type(layer) is nn.MaxUnpool2d:
        ind=self.indStack.pop()
        inp=layer(inp,ind)
      else:
        inp=layer(inp)
      #print(inp.shape)
      
      if type(layer) is nn.Linear:
        shape=self.linStack.pop()
        inp=inp.view(shape)
        
    return inp
  
  def forward(self,inp):
    inp=self.encode(inp)
    inp=self.decode(inp)
    return inp

model=Net(64)
model=model.to(device)
model=torch.nn.DataParallel(model)

criterion = torch.nn.MSELoss()
regul = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                weight_decay=0.0,
                                nesterov=True)
sched=torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.9, last_epoch=-1)

for epoch in range(1000):
  accloss=0
  counter=0
  for inp, target in loader:
    inp=inp.to(device)
    target=target.to(device)
    dic=model.state_dict()
    '''
    w=dic['conv.weight']
    v=dic['deconv.weight']
    w2=dic['conv2.weight']
    v2=dic['deconv2.weight']
    '''
    output=model(inp)
    loss=criterion(inp, output)
    accloss+=loss
    counter+=1
    '''
    1*(regul(w,torch.zeros_like(w))+regul(v,torch.zeros_like(v)))+\
    1*(regul(w2,torch.zeros_like(w2))+regul(v2,torch.zeros_like(v2)))
    '''
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  sched.step()
  print(epoch, accloss/counter)

"""# Test on unseen image
(Does not work now, input image must be resized)
"""

im=io.imread('drive/My Drive/superres/test2.png')
tim=batchTensor(im)

with torch.no_grad():
  out=model(tim)
  code=model.module.encode(tim)
#recon=batchTensor(recon)
imout=batchTensorToNP(out)
codeim=batchTensorToNP(code)

plt.imshow(im)

#Display encoded image (wont work if encoding is not an image)
#plt.imshow(codeim[:,:,0])

plt.imshow(imout)

"""#Visualize on training data"""

#pick an arbitrary image. Note: loader is randomized
batch_index=1
index_in_batch=0
for i, batch in enumerate(loader):
  if i==batch_index:
    break
inp=batch[0][index_in_batch,:,:,:]
inp=inp.unsqueeze(0)
inp=inp.to(device)

with torch.no_grad():
  outp=model(inp)
  cod=model.module.encode(inp)

plt.imshow(batchTensorToNP(inp))

#Display encoded image (wont work if encoding is not an image)
#plt.imshow(batchTensorToNP(cod)[:,:,0])

plt.imshow(batchTensorToNP(outp))

