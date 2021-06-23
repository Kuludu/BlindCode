import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.distributions import Normal, kl_divergence
from torchvision.utils import save_image
from torch.utils.data import Dataset, TensorDataset

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
def load_dataset() -> Dataset:
    print("数据集加载中....")
    data = np.load("data/mnist.npz")
    x_train, y_train = data["x_train"], data["y_train"]
    x_train = np.array(x_train, dtype=np.float) / 255.0
    np.random.shuffle(x_train)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    print("数据集加载完毕....")
    return TensorDataset(x_train, y_train)


# 超参数
image_size = 28 * 28
h_dim = 400
z_dim = 40
num_epochs = 40
batch_size = 64
learning_rate = 0.001

data_loader = torch.utils.data.DataLoader(dataset=load_dataset(),
                                          batch_size=batch_size,
                                          shuffle=True)


class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=40):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        x = self.fc1(x)
        h = F.tanh(x)
        return self.fc2(h), self.fc3(h)

    def reparamterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc4(z)
        h = F.tanh(h)
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparamterize(mu, log_var)
        reconst_mu = self.decode(z)
        return mu, log_var, reconst_mu

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


prior = Normal(0,1)
for epoch in range(num_epochs):
    for i, (x, _) in enumerate(data_loader):

        x = x.view(-1, image_size)
        mu, log_var, reconst_mu = model(x)
        reconst_loss = F.binary_cross_entropy(reconst_mu, x, size_average=False)
        kl_div = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)

        # print("kl_div 计算方式1：",kl_div)
        
        # pred = Normal(mu, torch.exp(log_var / 2))
        # kl_div = kl_divergence(pred, prior)
        # kl_div = kl_div.sum()
        # print("kl_div 计算方式2：",kl_div)

        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))
            
    with torch.no_grad():
        z = torch.randn(batch_size, z_dim)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(
            output_dir, 'sampled-{}.png'.format(epoch+1)))
        
        _, _, reconst_mu = model(x)
        x_concat = torch.cat(
            [x.view(-1, 1, 28, 28), reconst_mu.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(
            output_dir, 'reconst-{}.png'.format(epoch+1)))

print("模型保存完毕")
torch.save(model, 'model.pkl')
