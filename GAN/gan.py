import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets 
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

epochs = 1
lr = 0.0001

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                     std=(0.5, 0.5, 0.5))])
# MNIST size: 28x28
mnist = datasets.MNIST(root='./data/',
                       train=True,
                       transform=transform,
                       download=False)
# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=100, 
                                          shuffle=True)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.disc = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.LeakyReLU(0.1),
			nn.Linear(512, 512),
			nn.LeakyReLU(0.1),
			nn.Linear(512, 1),
			nn.Sigmoid())

	def forward(self, x):
		output = self.disc(x)
		return output

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.gen = nn.Sequential(
			nn.Linear(32, 512),
			nn.LeakyReLU(0.1),
			nn.Linear(512, 512),
			nn.LeakyReLU(0.1),
			nn.Linear(512, 28*28),
			nn.Tanh())

	def forward(self, x):
		output = self.gen(x)
		return output

d = Discriminator()
g = Generator()
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(d.parameters(), lr)
g_optimizer = torch.optim.Adam(g.parameters(), lr)


for e in range(epochs):
	for i, (images, _) in enumerate(data_loader):

		# batch size
		bz = images.size(0)

		images = Variable(images.view(bz, -1))

		# labels
		real_labels = Variable(torch.ones(bz))
		fake_labels = Variable(torch.zeros(bz))

		# ----------- TRAIN DISCRIMINATOR ----------

		# forward pass on discriminator with mnist data
		outputs = d(images)

		# computer loss for mnist data
		d_real_error = criterion(outputs, real_labels)
		real_score = outputs

		# generate images from a 28x32 random noise input
		input_noise = Variable(torch.randn(bz, 32))
		fake_images = g(input_noise)

		# run gan-generated images through discriminator and compute loss
		outputs = d(fake_images)
		d_fake_error = criterion(outputs, fake_labels)

		# ---------- BACKPROP ------------
		d_loss = d_real_error
		d.zero_grad()
		d_loss.backward()
		d_optimizer.step()

		# ------------ TRAIN GENERATOR --------------
		noise = Variable(torch.randn(bz, 32))
		fake_images = g(noise)
		outputs = d(fake_images)

		g_loss = criterion(outputs, real_labels)

		# ----------- BACKPROP -----------
		d.zero_grad()
		g.zero_grad()
		g_loss.backward()
		g_optimizer.step()

		
		print("epoch:", e, "g_loss:", g_loss.data[0])

# save images
fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
save_image(fake_images.data.clamp(0, 1), '/Users/tejpalvirdi/Desktop/gan-data/a.png')


