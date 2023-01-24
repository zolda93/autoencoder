from torch_snippets import*
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize,Lambda
from torchvision.utils import make_grid
device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5],std=[0.5]),
    Lambda(lambda x:x.to(device)),
    ])


training_data = MNIST('./content/',transform=image_transform,train=True,download=False)
testing_data = MNIST('./content/',transform=image_transform,train=False,download=False)

class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x.view(x.shape[0],32,4,4)


class CCVAE(nn.Module):
    def __init__(self,z_dim,class_size):
        super().__init__()
        

        self.train = DataLoader(training_data,batch_size=64,shuffle=True)
        self.test = DataLoader(testing_data,batch_size=64,shuffle=False)

        self._encoder = nn.Sequential(
                nn.Conv2d(2,16,5,stride=2),
                nn.ReLU(True),
                nn.Conv2d(16,32,5,stride=2),
                nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(32*4*4,300),
                nn.ReLU(True),)

        self.mean = nn.Sequential(nn.Linear(300,z_dim))
        self.std = nn.Sequential(nn.Linear(300,z_dim))

        self._decoder = nn.Sequential(
                nn.Linear(z_dim+class_size,300),
                nn.Linear(300,32*4*4),
                Reshape(),
                nn.ConvTranspose2d(32,16,5,stride=2),
                nn.ReLU(True),
                nn.ConvTranspose2d(16,1,5,stride=2),
                nn.ReLU(True),
                nn.ConvTranspose2d(1,1,4),
                nn.ReLU(True),)

        self._loss = nn.MSELoss(reduction='sum')
        self.optimizer = optim.AdamW(self.parameters(),lr=1e-3)


    def encoder(self,x,c):
        c = torch.argmax(c,dim=1).reshape(c.shape[0],1,1,1)
        c = torch.ones(x.shape).to(device)*c
        x = torch.cat((x,c),dim=1)
        x = self._encoder(x)
        return self.mean(x),self.std(x)

    def decoder(self,x):
        return self._decoder(x)

    def sampling(self,mean,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self,x,c):
        mean,log_var = self.encoder(x,c)
        z = self.sampling(mean,log_var)
        z = torch.cat((z,c.float()),dim=1)
        return self.decoder(z),mean,log_var

    def loss(self,recon_x,x,mean,log_var):
        recon = self._loss(recon_x,x)
        kld = -0.5*torch.sum(1+log_var - mean.pow(2) - log_var.exp())
        return recon + kld,recon,kld

    def train_batch(self,x,c):
        c = c.to(device)
        self.optimizer.zero_grad()
        recon,mean,log_var = self.forward(x,c)
        loss,mse,kld = self.loss(recon,x,mean,log_var)
        loss.backward()
        self.optimizer.step()
        return loss,mse,kld,log_var.mean(),mean.mean()

    @torch.no_grad()
    def validate_batch(self,x,c):
        c = c.to(device)
        recon,mean,log_var = self.forward(x,c)
        loss,mse,kld = self.loss(recon,x,mean,log_var)
        return loss,mse,kld,log_var.mean(),mean.mean()










