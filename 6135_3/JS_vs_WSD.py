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
import samplers as smplr

#Check cuda
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#To initialize weights of network
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
		
		
##############PROBLEM 1.1 
##########################
#Discriminator for Q1.1
class Discriminator(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, output_size)
  
  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return torch.sigmoid_(self.fc3(x))
	
#Q1.1 : Loss function to estimate JS Divergence
def js_loss(D, dist1, dist2):
  dist1_loss = torch.mean(torch.log(D(dist1)))
  dist2_loss = torch.mean(torch.log(1-D(dist2)))
  return -(dist1_loss+dist2_loss)
  
#Q1.1 : Train function to estimate JS Divergence using js_loss
def train_js(_phi = 0):
  # Model parameters
  d_input_size = 2    
  d_hidden_size = 10
  d_output_size = 1     
  minibatch_size = 512
  d_learning_rate = 1e-3
  sgd_momentum = 0.9
  num_epochs = 3000
  print_interval = 500
  dist1_sample, dist2_sample= None, None

  D = Discriminator(input_size=d_input_size,
                      hidden_size=d_hidden_size,
                      output_size=d_output_size)
  
  if cuda:
    D.cuda()
    
  D.apply(init_weights) #Initialize discriminator weights
  d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum) #Optimizer SGD with momentum
  
  dist1 = iter(smplr.distribution1(0, 512)) #Distribution1= (0, Z) iterator where Z~Uniform(0,1)
  dist2 = iter(smplr.distribution1(_phi, 512)) #Distribution2 = (_phi, Z)iterator where Z~Uniform(0,1)

  for epoch in range(num_epochs):
    # Initialize grads
    D.zero_grad()
    
    #Generate dist1 data
    dist1_sample = next(dist1)
    if cuda:
      dist1_sample = torch.from_numpy(dist1_sample).float().cuda()
      dist1_sample = Variable(dist1_sample)
      #print(dist1_sample.shape)

    #Generate dist2 data
    dist2_sample = next(dist2)
    if cuda:
      dist2_sample = torch.from_numpy(dist2_sample).float().cuda()
      dist2_sample = Variable(dist2_sample)
      #print(dist2_sample.shape)

    #Calculate loss
    loss = js_loss(D, dist1_sample, dist2_sample)
    loss.backward()

    d_optimizer.step() #  optimizes D's parameters; changes based on stored gradients from backward()
    

    if epoch % print_interval == 0:
      print("Epoch %s: D ( JS Loss %s ) " %(epoch, loss))
      
    #print("Epoch %s: D JS Loss %s  " %(epoch, loss))
   
  #Return final loss
  return loss

##############PROBLEM 1.2
##########################
#Q1.2 Critic Network
class Critic(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, output_size)
  
  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)

#Q1.2 Loss function to estimate Wasserstein Distance
def wd_loss(critic, dist1_samples, dist2_samples, lamda =10):
  dist1_loss = torch.mean(critic(dist1_samples))
  dist2_loss = torch.mean(critic(dist2_samples))
  a = torch.FloatTensor(512, 1).uniform_(0, 1)
  a = a.float().cuda()
  z = a * dist1_samples + (1-a)*dist2_samples
  z = Variable(z, requires_grad = True)
  T_z  = critic(z)
  grad_penalty = grad( T_z , z, grad_outputs=torch.ones(T_z.size()).cuda(),create_graph=True, only_inputs=True, retain_graph=True)[0]
  grad_penalty = grad_penalty.view(512, -1)
  grad_penalty = (grad_penalty.norm(2, dim=1) -1)**2
  wd_loss = -(dist1_loss - dist2_loss -  lamda * grad_penalty.mean())
  return wd_loss
  
#Q1.2 : Train function to estimate Wasserstein Distance using wd_loss
def train_wd(_phi = 0):
  # Model parameters
  d_input_size = 2    
  d_hidden_size = 10   
  d_output_size = 1     
  minibatch_size = 512
  d_learning_rate = 1e-3
  sgd_momentum = 0.9
  num_epochs = 3000
  print_interval = 500
  dist1_sample, dist2_sample= None, None

  T = Critic(input_size=d_input_size, 
             hidden_size=d_hidden_size, 
             output_size=d_output_size)
  if cuda:
    T.cuda()
    
  T.apply(init_weights) #Initialize critic
  
  d_optimizer = optim.SGD(T.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
  
  dist1 = iter(smplr.distribution1(0, 512)) #Distribution1= (0, Z) iterator where Z~Uniform(0,1)
  dist2 = iter(smplr.distribution1(_phi, 512)) #Distribution2 = (_phi, Z)iterator where Z~Uniform(0,1)
  
  for epoch in range(num_epochs):
    # Initialize grads
    T.zero_grad()
    
    #Generate dist1 data
    dist1_sample = next(dist1)
    if cuda:
      dist1_sample = torch.from_numpy(dist1_sample).float().cuda()
      dist1_sample = Variable(dist1_sample)
      #print(dist1_sample.shape)

    #Generate dist2 data
    dist2_sample = next(dist2)
    if cuda:
      dist2_sample = torch.from_numpy(dist2_sample).float().cuda()
      dist2_sample = Variable(dist2_sample)
      #print(dist2_sample.shape)

    #Calculate loss
    loss = wd_loss(T, dist1_sample, dist2_sample)
    loss.backward()
    
    d_optimizer.step() #  optimizes D's parameters; changes based on stored gradients from backward()

    if epoch % print_interval == 0:
      print("Epoch %s: D (%s WD Loss) " %(epoch, loss))
  
  #Return final loss
  return loss

##############PROBLEM 1.3
##########################
#Q1.3 - Jenson Shannon divergence calculation
jsd_loss_list = []
for i in np.linspace(-1,1,21):
  _phi = i
  print('_phi = %s' %_phi)
  train_loss = train_js(_phi)
  jsd_loss =  np.log(2) - train_loss.item()/2.0 
  jsd_loss_list.append(jsd_loss)
  
#Q1.3 Wasserstein Distance calculation
wd_loss_list = []
for i in np.linspace(-1,1,21):
  _phi = i
  print('_phi = %s' %_phi)
  train_loss = train_wd(_phi)
  wd_loss_ =  - train_loss.item() 
  wd_loss_list.append(wd_loss_)
  
#Q1.3 Plot JS Divergence vs Wasserstein Distance
x = np.linspace(-1,1,21)
plt.plot(x, jsd_loss_list)
plt.plot(x, wd_loss_list)
plt.legend(['J-S Divergence','Wasserstein Distance'])
plt.title('J-S Divergence vs Wasserstein Distance')
plt.show()
