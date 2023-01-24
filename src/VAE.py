from torch_snippets import*
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torchvision.transforms import Compose,ToTensor,Normalize,Lambda

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5],std=[0.5]),
    Lambda(lambda x:x.to(device)),
    ])


training_data = MNIST('./content/',transform=image_transform,train=True,download=False)
testing_data = MNIST('./content/',transform=image_transform,train=False,download=False)



class Reshape(nn.Module):
    def __init__(self,shape):
        self.shape = shape
        super().__init__()

    def forward(self,x):
        return x.view(*self.shape)



class ConvolutionVAE(nn.Module):
    def __init__(self,h_dim,z_dim):
        super().__init__()

        self.train = DataLoader(training_data,batch_size=64,shuffle=True)
        self.test = DataLoader(testing_data,batch_size=64,shuffle=False)

        self._encoder = nn.Sequential(
                nn.Conv2d(1,32,3,stride=3,padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,stride=2,padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2,stride=1),
                nn.Flatten(),
                nn.Linear(256,h_dim))

        self.mean = nn.Sequential(nn.Linear(h_dim,z_dim))
        self.std = nn.Sequential(nn.Linear(h_dim,z_dim))

        self._decoder = nn.Sequential(
                nn.Linear(z_dim,256),
                Reshape((-1,64,2,2)),
                nn.ConvTranspose2d(64,32,3,stride=2),
                nn.ReLU(True),
                nn.ConvTranspose2d(32,16,5,stride=3,padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(16,1,2,stride=2,padding=1),
                nn.Tanh(),)
        
        self._loss = nn.MSELoss(reduction='sum')
        self.optimizer = optim.AdamW(self.parameters(),lr=1e-3)

    def encoder(self,x):
        x = self._encoder(x)
        return self.mean(x),self.std(x)
    
    def decoder(self,x):
        return self._decoder(x)

    def sampling(self,mean,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self,x):
        mean,log_var = self.encoder(x)
        z = self.sampling(mean,log_var)
        return self.decoder(z),mean,log_var

    def loss(self,recon_x,x,mean,log_var):
        recon = self._loss(recon_x,x)
        kld = -0.5*torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return recon + kld,recon,kld

    def train_batch(self,x):
        self.optimizer.zero_grad()
        recon,mean,log_var = self.forward(x)
        loss,mse,kld = self.loss(recon,x,mean,log_var)
        loss.backward()
        self.optimizer.step()
        return loss,mse,kld,log_var.mean(),mean.mean()

    @torch.no_grad()
    def validate_batch(self,x):
        recon,mean,log_var = self.forward(x)
        loss,mse,kld = self.loss(recon,x,mean,log_var)
        return loss,mse,kld,log_var.mean(),mean.mean()






