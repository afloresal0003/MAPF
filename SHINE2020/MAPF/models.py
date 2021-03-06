import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Flatten(nn.Module):
    """
    A 'layer' that flattens the input with batches, can be used in a nn.Sequential
    """

    def forward(self, x):
        """
        x is a b x h x w x ..., (b times any number of dimensions)
        returns an array of b x n (flattens everything after the batch dimension)
        """
        return x.view(x.size()[0], -1)

class ConvNet(nn.Module):
    def __init__(self,  in_channels=3, width=11, height=11):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        # self.max_pool = nn.MaxPool2d(2)
        self.linear = nn.Linear(int(16 * (width) * (height) ), 32)
        self.linear2 = nn.Linear(32, 32)
        self.actor_linear = nn.Linear(32, 5)
        self.critic_linear = nn.Linear(32, 1)
        self.saved_actions = []
        self.rewards = []
        self.net = nn.Sequential(
                    self.conv1,
                    self.relu,
                    self.conv2,
                    self.relu,
        #            self.max_pool,
                    Flatten(),
                    self.linear,
                    self.relu,
                    self.linear2,
                    self.relu
                    )

    def forward(self, x, eps=.0001):
        """
        x is a tuple of image observation (including obstacles, agents, goals, etc...) and a goal vector (x, y to goal) for each agent
        """

        # Turn x into tensors for observation and goal vectors
        x = np.array(x)
        x, goal_vector = x[:, 0], x[:, 1]
        x = np.array([np.array(x_, dtype=int) for x_ in x])
        goal_vector = np.array([np.array(g) for g in goal_vector])
        x = torch.tensor(x).float()
        goal_vector =  torch.tensor(goal_vector).float()

        # Old code for if image data was in a batch or just an image
        # If image is fed in alone, add an empty batch dimension with unsqueeze
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # Run network
        x = self.net(x)

        # Calculate action probabilities, softmax converts numbers to probabilities (makes them sum to 1)
        # Empirically, Softmax performs better than just dividing by the sum of the numbers
        action_prob = F.softmax(self.actor_linear(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.critic_linear(x)

        # Make sure the actions actually sum to 1, add epsilon to avoid dividing by 0
        # When we use these later, we get errors if we have a 0 probability of taking an action
        action_prob = action_prob + eps
        action_prob /= action_prob.sum()

        return action_prob, state_values


class Policy(nn.Module):
    """
    implements both actor and critic in one model, old code from cartpole
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 12)
        # actor's layer
        self.action_head = nn.Linear(12, 5)
        # critic's layer
        self.value_head = nn.Linear(12, 1)
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []


    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)
        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)
        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        action_prob = action_prob + .001
        action_prob /= action_prob.sum()
        return action_prob, state_values

class Primal(nn.Module):
    """
    An approximation of the primal model (without the LSTM, or some auxilary losses they use)
    It's big, but quick to code up, look back at the paper, try and find what's changed and how the image gets translated into code
    """
    RNN_SIZE = 512

    def __init__(self,  in_channels=3, width=11, height=11):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels, self.RNN_SIZE // 4, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv1_2 = nn.Conv2d(self.RNN_SIZE // 4, self.RNN_SIZE // 4, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv1_3 = nn.Conv2d(self.RNN_SIZE // 4, self.RNN_SIZE // 4, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.max_pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(self.RNN_SIZE // 4, self.RNN_SIZE // 2, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv2_2 = nn.Conv2d(self.RNN_SIZE // 2, self.RNN_SIZE // 2, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv2_3 = nn.Conv2d(self.RNN_SIZE // 2, self.RNN_SIZE // 2, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        # Max Pool
        self.conv3 = nn.Conv2d(self.RNN_SIZE // 2, self.RNN_SIZE // 4, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.flatten = Flatten()
        self.goal_linear = nn.Linear(2, 12)
        self.linear1 = nn.Linear(self.RNN_SIZE + 12, self.RNN_SIZE)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.RNN_SIZE, self.RNN_SIZE)

        self.actor = nn.Linear(self.RNN_SIZE, 5)
        self.critic = nn.Linear(self.RNN_SIZE, 1)

        self.convs = nn.Sequential(
                    self.conv1_1,
                    self.conv1_2,
                    self.conv1_3,
                    self.max_pool1,
                    self.conv2_1,
                    self.conv2_2,
                    self.conv2_3,
                    self.max_pool1,
                    self.conv3,
                    self.flatten
                )

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = np.array(x)
        x, goal_vector = x[:, 0], x[:, 1]
        x = np.array([np.array(x_, dtype=int) for x_ in x])
        goal_vector = np.array([np.array(g) for g in goal_vector])
        x = torch.tensor(x).float()
        goal_vector =  torch.tensor(goal_vector).float()

        x = self.convs(x)
        convs = x
        g = self.goal_linear(goal_vector)
        g = self.relu(g)
        x = torch.cat((x, g), dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        x_actor = self.actor(x)
        state_values = self.critic(x)

        action_prob = nn.Softmax(dim=1)(x_actor)

        action_prob = action_prob + .001
        action_prob /= action_prob.sum()
        return action_prob, state_values


class DeepConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.flatten = Flatten()
        self.max_pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(100, 25, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.linear1 = nn.Linear(25 * 5 * 5, 128)
        self.relu = nn.ReLU()
        self.linear_actor = nn.Linear(130, 5)
        self.linear_critic = nn.Linear(130, 1)

        self.net = nn.Sequential(
                    self.conv1,
                    self.max_pool1,
                    self.conv2,
                    self.relu,
                    self.flatten,
                    self.linear1,
                    self.relu,
                )

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = np.array(x)
        x, goal_vector = x[:, 0], x[:, 1]
        x = np.array([np.array(x_, dtype=int) for x_ in x])
        goal_vector = np.array([np.array(g) for g in goal_vector])
        x = torch.tensor(x).float()
        goal_vector =  torch.tensor(goal_vector).float()

        x = self.net(x)
        x = torch.cat((x, goal_vector), dim=1)

        x_actor = self.linear_actor(x)
        state_values = self.linear_critic(x)

        action_prob = nn.Softmax(dim=1)(x_actor)

        action_prob = action_prob + .001
        action_prob /= action_prob.sum()

        return action_prob, state_values
