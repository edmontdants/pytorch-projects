import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

dtype = torch.FloatTensor

def sigmoid(x):
  return 1/(1 + math.e ** -x)

d_in = 3
hidden1 = 5
hidden2 = 5
d_out = 1
lr = 0.001
epochs = 100

a = np.array([[5,1,10], [8,3,4], [9,7,1], [5,2,8], [8,4,3]])
x = Variable(torch.from_numpy(a).type(dtype), requires_grad = False)

b = np.array([50,96,63,80,96])
y = Variable(torch.from_numpy(b).type(dtype), requires_grad = False)

w1 = Variable(torch.randn(d_in, hidden1).type(dtype), requires_grad = True)
w2 = Variable(torch.randn(hidden1, hidden2).type(dtype), requires_grad = True)
w3 = Variable(torch.randn(hidden2, d_out).type(dtype), requires_grad = True)


for i in range(epochs):
  output = sigmoid(x.mm(w1))
  output = sigmoid(output.mm(w2))
  output = sigmoid(output.mm(w3))
  
  # mean squared loss:
  loss = (output - y).pow(2).sum()
  
  print("Epoch:", i, "  Loss:", loss.data[0]/100)

  loss.backward()

  w1.data -= lr * w1.grad.data
  w2.data -= lr * w2.grad.data
  w3.data -= lr * w3.grad.data

  w1.grad.data.zero_()
  w2.grad.data.zero_()
  w3.grad.data.zero_()

print(x.mm(w1).mm(w2).mm(w3))
