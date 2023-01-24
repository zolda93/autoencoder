import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch_snippets import*
from torchvision import transforms,datasets


device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5),std=(0.5))
    ])


training_data = datasets.MNIST('./content/',transform=image_transform,train=True,download=True)
testing_data = datasets.MNIST('./content/',transform=image_transform,train=False,download=False)


class AutoEncoder(nn.Module):
    def __init__(self,latent_dim,batch_size,lr):
        super().__init__()

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr

        self.train = DataLoader(training_data,batch_size=self.batch_size,shuffle=True)
        self.test = DataLoader(testing_data,batch_size=self.batch_size,shuffle=False)

        self.encoder = nn.Sequential(
                nn.Linear(28*28,64),
                nn.ReLU(),
                nn.Linear(64,32),
                nn.Linear(32,self.latent_dim))
        
        self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim,32),
                nn.ReLU(),
                nn.Linear(32,64),
                nn.ReLU(),
                nn.Linear(64,28*28),
                nn.Tanh())

        self.optimizer = optim.Adam(self.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()

    def forward(self,x):
        x = x.view(len(x),-1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(len(x),1,28,28)
        return x

    def train_batch(self,x):

        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.loss(output,x)
        loss.backward()
        self.optimizer.step()
        return loss
    
    @torch.no_grad()
    def validate_batch(self,x):
        output = self.forward(x)
        loss = self.loss(output,x)
        return loss



