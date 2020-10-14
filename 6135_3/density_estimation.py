from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt
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

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))

############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######

#Discriminator
class Discriminator_1(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Discriminator_1, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, output_size)
  
  def forward(self,x):
    x = F.leaky_relu_(self.fc1(x), negative_slope=0.1)
    x = F.leaky_relu_(self.fc2(x), negative_slope=0.1)
    return  torch.sigmoid(self.fc3(x))
  #def forward(self,x):
  #  x = torch.sigmoid(self.fc1(x))
  #  x = torch.sigmoid(self.fc2(x))
  #  return  torch.sigmoid(self.fc3(x))

D = Discriminator_1(input_size=1,
                  hidden_size=10,
                  output_size=1)
if cuda:
  D.cuda()
D.apply(init_weights)

d_learning_rate = 1e-3
#sgd_momentum = 0.9
num_epochs =  5000
print_interval = 1000
f0_sample, f1_sample= None, None
d_optimizer = optim.Adam (D.parameters(), lr=d_learning_rate)#, momentum=sgd_momentum)

dist_gaussian = iter(smplr.distribution3(512))
dist_unknown = iter(smplr.distribution4(512))

for epoch in range(num_epochs):
  # Initialize grads
  D.zero_grad()
    
  #Generate dist1 data
  f0_sample = next(dist_gaussian)
  if cuda:
    f0_sample = torch.from_numpy(f0_sample).float().cuda()
    f0_sample = Variable(f0_sample)
    #print(f0_sample.shape)

  #Generate dist2 data
  f1_sample = next(dist_unknown)
  if cuda:
    f1_sample = torch.from_numpy(f1_sample).float().cuda()
    f1_sample = Variable(f1_sample)
    #print(f1_sample.shape)
    
  #Calculate loss
  f0_loss = torch.mean(torch.log(1-D(f0_sample)))
  f1_loss = torch.mean(torch.log(D(f1_sample)))
  loss = -(f0_loss + f1_loss)
  loss.backward()

  d_optimizer.step() #  optimizes D's parameters; changes based on stored gradients from backward()
   
  if epoch % print_interval == 0:
    print("Epoch %s: D ( Loss %0.4f ) " %(epoch, loss.item()))
	

D.eval()
xx = np.linspace(-5,5,1000)
xx_tensor = torch.from_numpy(xx).float().cuda()
xx_tensor = xx_tensor.view(1000,1)
D_xx = D(xx_tensor)  #Output of discriminator

xx_tensor = xx_tensor.cpu()
D_xx = D_xx.cpu()

############### plotting things
############### (1) plot the output of your trained discriminator 
r = D_xx.detach().numpy()
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx, r)
plt.title(r'$D(x)$')

############### (2) plot the estimated density contrasted with the true density
estimated_f1 = N(xx_tensor) * D_xx  / (1-D_xx)
estimate = estimated_f1.detach().numpy() 
#PLot
#plt.subplot(1,2,2)
plt.plot(xx, estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')