'''Single agent path finding
Made gym
Made the network
Then trained it
Lastly, debugged (Hyperparemeter tuning)
    - Neural Network restructuring
    - Reward shaping
    - Algorithmn Configuration'''

import numpy as np
import gym
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from itertools import count
from collections import namedtuple
import torch.nn.functional as F
from torch.distributions import Categorical

action_dict = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}

# Savedaction contains the logprob of taking an action, and the value
# This is essentially [the log of the actor output, the critic output]
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Give me some torch error detection please
torch.autograd.set_detect_anomaly(True)

class PFenv(gym.Env):

    def __init__(self, max_x, max_y):
    def __init__(self, max_x, max_y, prob_obs = 0.5):
        self.max_x = max_x
        self.max_y = max_y
        self.prob_obs = prob_obs
        self.state = np.zeros((self.max_x, self.max_y))
        for row in range(len(self.state)):
            for i in range(len(self.state)):
                if (random.random()) < prob_obs:
                    self.state[row, i] = -1
        self.agent = (np.random.randint(self.max_x),
                      np.random.randint(self.max_y))
        self.goal = (np.random.randint(self.max_x),
                     np.random.randint(self.max_y))
        self.state[self.agent] = 1
        self.state[self.goal] = 2

    def reset(self):
        self.state = np.zeros((self.max_x, self.max_y))
        for row in range(len(self.state)):
            for i in range(len(self.state)):
                if (random.random()) < self.prob_obs:
                    self.state[row, i] = -1
        self.agent = 0
        self.goal = 0

        while(self.agent == self.goal):
            self.agent = (np.random.randint(self.max_x),
                          np.random.randint(self.max_y))
            self.goal = (np.random.randint(self.max_x),
                         np.random.randint(self.max_y))

        self.state[self.agent] = 1
        self.state[self.goal] = 2
        self.agent = np.array(self.agent)

        return np.array(self.goal) - self.agent

        return self.get_observation(self.state)
    def render(self):
        print(self.state)

    def get_observation(self, state):
        obstacles = np.where(state==-1, 1, 0)
        goal = np.where(state==2, 1, 0)
        agent = np.where(state==1, 1, 0)
        return np.stack((obstacles, goal, agent))
    def step(self, action):
        direction = action_dict[action]
        if self.is_valid_action(action) == True:
@@ -60,79 +65,95 @@ def step(self, action):
        if (self.agent == self.goal).all():
            done = True
            reward = 50
        #if (self.agent == ) #Conditional to test whether or not the agent has hit an obstacle, then give negative reward if so.
        else:
            done = False
            reward = -1
        obs = self.goal - self.agent  # an x, y pair
        obs = self.get_observation(self.state)  # an x, y pair
        return obs, reward, done, None

    def is_valid_action(self, action):
        action_dict = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
        direction = action_dict[action]
        loc = self.agent + np.array(direction)
        if loc[0] >= 0 and loc[0] < self.max_x:
            if loc[1] >= 0 and loc[1] < self.max_y:
                return True
                if self.state[loc[0], loc[1]] != -1:
                    return True
        return False

    def close(self):
        pass

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(-1)

class convNet(nn.Module):
    def __init__(self, maxX, maxY):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.actor_linear = nn.Linear(16*maxX*maxY, 5)
        self.critic_linear = nn.Linear(16*maxX*maxY, 1)
        self.saved_actions = []
        self.rewards = []
        self.net = nn.Sequential(
                    self.conv1,
                    self.relu,
                    self.conv2,
                    self.relu,
                    Flatten(),
                    )
    def forward(self,x):
        x = x.unsqueeze(0)
        x = self.net(x)
        action_prob = F.softmax(self.actor_linear(x), dim=-1)
        # critic: evaluates being in the state s_t
        state_values = self.critic_linear(x)
        action_prob = action_prob + .001
        action_prob /= action_prob.sum()
        return action_prob, state_values
class Policy(nn.Module):
    """
    implements both actor and critic in one model
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

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer

    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()

def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
@@ -148,100 +169,96 @@ def finish_episode():
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # This breaks us... I think it's when you get to the goal on the first step
    # Easiest solution is to just skip it
    if len(returns) == 1:
        del model.rewards[:]
        del model.saved_actions[:]
        return

        return 0,0
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() #list of policy losses
    # perform backprop
    loss.backward()

    # nn.utils.clip_grad_norm_(model.parameters(), 1)

    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

model = Policy()
    return (torch.stack(policy_losses).mean().detach()), (torch.stack(value_losses).mean().detach())
#model = Policy()
model = convNet(4,4)
clip_value = 10
optimizer = optim.Adam(model.parameters(), lr=3e-2)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()

writer = SummaryWriter()
def main(env):
    running_reward = 10
    log_interval = 1
    log_interval = 5
    # run inifinitely many episodes
    policyLossList = []
    valueLossList =[]
    rewardL = []
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 100):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            if False:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        policyLoss, valueLoss = finish_episode()
        policyLossList.append(policyLoss)
        valueLossList.append(valueLoss)
        rewardL.append(running_reward)
        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

            writer.add_scalar("Reward", running_reward, i_episode)
            writer.add_scalar("VLoss", valueLoss, i_episode)
            writer.add_scalar("PLoss", policyLoss, i_episode)
            writer.add_scalar("Loss", policyLoss + valueLoss, i_episode)
        # check if we have "solved" the cart pole problem
        if running_reward > 45:
        if running_reward > 40:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


    plt.plot(policyLossList, label = "Policy")
    plt.plot(valueLossList, label = "Value")
    plt.legend()
    plt.show()
    plt.plot(rewardL, label = "Reward")
    plt.show()
if __name__ == "__main__":
    max_x = 4
    max_y = 4
    env = PFenv(max_x, max_y)
    env = PFenv(max_x, max_y, 0.2)
    main(env)
    env.render()
    print(env.get_observation(env.state).shape)
'''
    env = PFenv(max_x, max_y)
    env.render()
    env.reset()
