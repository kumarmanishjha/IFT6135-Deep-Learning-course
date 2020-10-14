# -*- coding: utf-8 -*-
"""Copy of Asgn3_P1_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fMsJGVUM2DyOt2WrQVMEbYk3_4ZFfU9c
"""

def kl_divergence(p,q):
  epsilon = 0.000001
  p = p+epsilon
  q = q+epsilon
  kl_d = np.sum(p*np.log(p/q))
  return kl_d

def js_divergence(p,q):
  r = (p+q)/2
  js_d = kl_divergence(p,r)/2 +  kl_divergence(q,r)/2
  return js_d

def distribution1(x, batch_size=512):
    # Distribution defined as (x, U(0,1)). Can be used for question 3
    while True:
        yield(np.array([(x, random.uniform(0, 1)) for _ in range(batch_size)]))


def distribution2(batch_size=512):
    # High dimension uniform distribution
    while True:
        yield(np.random.uniform(0, 1, (batch_size, 2)))


def distribution3(batch_size=512):
    # 1D gaussian distribution
    while True:
        yield(np.random.normal(0, 1, (batch_size, 1)))

e = lambda x: np.exp(x)
tanh = lambda x: (e(x) - e(-x)) / (e(x)+e(-x))
def distribution4(batch_size=1):
    # arbitrary sampler
    f = lambda x: tanh(x*2+1) + x*0.75
    while True:
        yield(f(np.random.normal(0, 1, (batch_size, 1))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import math
from torch.autograd import Variable,grad

def js_loss(D, dist1, dist2):
  dist1_loss = torch.mean(torch.log(D(dist1)))
  dist2_loss = torch.mean(torch.log(D(dist2)))
  return -(dist1_loss+dist2_loss)

class Discriminator(nn.Module):
  def __init__(self, input_size, hidden_size, output_size,f):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, output_size)
    self.f = f
  
  def forward(self,x):
    #print(x.shape)
    x = self.f(self.fc1(x))
    #print(x.shape)
    x = self.f(self.fc2(x))
    #print(x.shape)
    return self.f(self.fc3(x))

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(_phi = 0):
  # Model parameters
  d_input_size = 2    # Minibatch size
  d_hidden_size = 10    # Discriminator hidden_size
  d_output_size = 1     # Single dimension for 'real' vs. 'fake' classification
  minibatch_size = 512

  d_learning_rate = 1e-3
  sgd_momentum = 0.9
  num_epochs = 5000
  print_interval = 100

  dfe, dre = 0, 0
  dist1_sample, dist2_sample= None, None

  discriminator_activation_function = torch.sigmoid
  D = Discriminator(input_size=d_input_size,
                      hidden_size=d_hidden_size,
                      output_size=d_output_size,
                      f=discriminator_activation_function)
  
  if cuda:
    D.cuda()
    
  #criterion = nn.BCELoss()  # Binary cross entropy
  d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)

  for epoch in range(num_epochs):
    # Initialize grads
    D.zero_grad()
    #Generate dist1 data
    dist1 = iter(distribution1(_phi, 512))
    dist1_sample = next(dist1)

    if cuda:
      dist1_sample = torch.from_numpy(dist1_sample).float().cuda()

    #  : Train D on dist_1
    #dist1_sample = Variable(d_sampler_1(d_input_size))
    #dist1_decision = D(dist1_sample)
    #dist1_error = criterion(dist1_decision, Variable(torch.ones([1,1])))  # ones = true
    #dist1_error.backward() # compute/store gradients, but don't change params

    #Generate dist2 data
    dist2 = iter(distribution2(512))
    dist2_sample = next(dist2)

    if device == 'cuda':
      dist2_sample = torch.from_numpy(dist2_sample).float().cuda()

    #  : Train D on dist_2
    #dist2_sample = Variable(d_sampler_2(d_input_size))
    #dist2_decision = D(dist2_sample)
    #dist2_error = criterion(dist2_decision, Variable(torch.zeros([1,1])))  # zeros = fake
    #dist2_error.backward()
    
    loss = js_loss(D, dist1_sample, dist2_sample)
    loss.bacward()

    d_optimizer.step() # Only optimizes D's parameters; changes based on stored gradients from backward()

    if epoch % print_interval == 0:
      print("Epoch %s: D (%s JS Loss) " %(epoch, loss))

train(0.1)

#1.2

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

dist1 = iter(distribution1(0, 512))
samples_dist1 = next(dist1)
samples_dist1.shape
samples_dist1 = torch.from_numpy(samples_dist1).float().to(device)
dist2 = iter(distribution1(0.1, 512))
samples_dist2 = next(dist2)
samples_dist2.shape
samples_dist2 = torch.from_numpy(samples_dist2).float().to(device)

def wd_loss(critic, dist1_samples, dist2_samples, lamda =10):
  dist1_loss = torch.mean(critic(dist1_samples))
  dist2_loss = torch.mean(critic(dist2_samples))
  
  a = torch.empty(512,1).uniform(0,1)
  a = a.float().cuda()
  z = a * dist1_samples + (1-a)*dist2_samples
  z = Variable(z, requires_grad = True)
  T_z  =D(z)
  grad_penalty = grad( T_z , z, torch.ones(T_z.size()).cuda() )[0]
  grad_penalty = grad_penalty.view(512, -1)
  grad_penalty = (grad_penalty.norm(2, dim=1) -1)**2
  wd_loss = -(dist1_loss - dist2_loss -  lamda * grad_penalty.mean())
  return wd_loss

critic =  Discriminator(input_size=d_input_size,
                      hidden_size=d_hidden_size,
                      output_size=d_output_size,
                      f=discriminator_activation_function)
if cuda:
    critic = critic.cuda()

n_iter = 1000
for iteration in xrange(n_iter):
  D.zero_grad()
  
  for iter_d in xrange(CRITIC_ITERS):
    #Generate dist2 samples
    dist1 = iter(distribution1(0, 512))
    dist1_samples = next(dist1)
    
    if device == 'cuda':
      dist1_sample = dist1_sample.cuda()

    dist1_samples = torch.from_numpy(dist1_samples).float().to(device)\
    
    #Train with dist1
    dist1_decision = D( Variable(dist1_samples))
    dist1_decision = dist1_decision.mean()
    dist1_decision.backward(mone)
    
    #Generate dist2 samples
    dist2 = iter(distribution2(_phi, 512))
    dist2_samples = next(dist2)
    
    if device == 'cuda':
      dist2_sample = dist2_sample.cuda()

    dist2_samples = torch.from_numpy(dist2_samples).float().to(device)\
    
    #Train with dist1
    dist2_decision = D( Variable(dist2_samples))
    dist2_decision = dist2_decision.mean()
    dist2_decision.backward(one)

    # train with gradient penalty
    gradient_penalty = compute_gradient_penalty(critic, Variable(dist1_sample), Variable(dist2_sample))
    gradient_penalty.backward()

    D_cost = D_fake - D_real + gradient_penalty
    Wasserstein_D = D_real - D_fake
    optimizerD.step()

#PLot for Q1.3
js_d_list = []
for i in np.linspace(-1,1,21):
  _phi = i
  js_d_list.append(train(_phi,js_d))
  
wd_list = []
for i in np.linspace(-1,1,21):
  _phi = i
  wd_list.append(train(_phi, wd))
  
#Plot
plt.plot(np.linspace(-1,1,21), js_d_list)
plt.annotate('JS_divergence')
plt.plot(np.linspace(-1,1,21), wd_list)
plt.annotate('Wasserstein Distance')
plt.show()

