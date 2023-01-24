from torch_snippets import*
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,Normalize,ToTensor,Lambda

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5],std=[0.5]),
    Lambda(lambda x: x.to(device)),
    ])

training_data = MNIST('./content/',transform=image_transform,train=True,download=False)
testing_data = MNIST('./content/',transform=image_transform,train=False,download=False)


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(1,32,3,stride=3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,stride=2),
                nn.Conv2d(32,64,3,stride=2,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,stride=1),)

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64,32,3,stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32,16,5,stride=3,padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16,1,2,stride=2,padding=1),
                nn.Tanh(),)

        self.loss = nn.MSELoss()
        self.optimizer = optim.AdamW(self.parameters(),lr=0.001,weight_decay=1e-5)

        self.train = DataLoader(training_data,batch_size=64,shuffle=True)
        self.test = DataLoader(testing_data,batch_size=64,shuffle=True)

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
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








