import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class ConvQNet(nn.Module):
    def __init__(self, obs_size=(20, 20), n_channels=3, n_actions=5):
        super(ConvQNet, self).__init__()
        self.obs_size=obs_size
        self.n_channels=n_channels
        self.n_actions=n_actions

        self.init_model()
        
    def init_model(self):
        self.conv1 = nn.Conv2d(self.n_channels, 32, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU()
        self.linear = nn.Linear(16*5*5, 5)

        self.net = nn.Sequential(
                self.conv1,
                self.relu,
                self.conv2,
                self.relu,
                self.conv3,
                self.relu,
                self.max_pool,
                self.conv4,
                self.relu,
                self.conv5,
                self.relu,
                self.conv6,
                self.relu,
                self.max_pool,
                Flatten(),
                self.linear
                )

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, obs_size=(20,20), n_channels=3, n_actions=5):
        super().__init__()
        self.model = nn.Sequential(
                nn.conv2d(self.n_channels, 32, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
                nn.ReLU(),
                nn.conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                Flatten(),
                nn.Linear(16*obs_size[0]/2*obs_size[1]/2),
                nn.Softmax()
            )
    def forward(self, x):
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, state_dim=(10,10)):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    net = ConvQNet()
    input = torch.rand(7, 3, 20, 20)
    out = net(input)
    print(out.shape)
