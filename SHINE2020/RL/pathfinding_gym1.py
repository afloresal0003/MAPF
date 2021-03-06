import numpy as np
import gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from itertools import count
from collections import namedtuple
import torch.nn.functional as F
from torch.distributions import Categorical

action_dict = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}

SavedAction = namedtuple('SavedAction', ['log_prob', 'value']) #come back to this

class PFenv(gym.Env):

    def __init__(self, max_x, max_y):
        self.max_x = max_x
        self.max_y = max_y
        self.state = np.zeros((self.max_x, self.max_y))
        self.agent = (np.random.randint(self.max_x),
                      np.random.randint(self.max_y))
        self.goal = (np.random.randint(self.max_x),
                     np.random.randint(self.max_y))
        self.state[self.agent] = 1
        self.state[self.goal] = 2

    def reset(self):
        self.state = np.zeros((self.max_x, self.max_y))
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

    def render(self):
        print(self.state)

    def step(self, action):
        direction = action_dict[action]
        if self.is_valid_action(action) == True:
            self.state[self.agent[0], self.agent[1]] = 0
            self.agent += np.array(direction)
            self.state[self.agent[0], self.agent[1]] = 1
        if (self.agent == self.goal).all():
            done = True
            reward = 50
        else:
            done = False
            reward = -1
        obs = self.goal - self.agent  # an x, y pair
        return obs, reward, done, None

    def is_valid_action(self, action):
        action_dict = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
        direction = action_dict[action]
        loc = self.agent + np.array(direction)
        if loc[0] >= 0 and loc[0] < self.max_x:
            if loc[1] >= 0 and loc[1] < self.max_y:
                return True
        return False

    def close(self):
        pass

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

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
        return action_prob, state_values

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()

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
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values
    gamma = 0.995
    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]: #iterating through backwards (::-1), kinda like step
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

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
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

def main(env):
    running_reward = 10
    log_interval = 1
    # run inifinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

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

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > 40:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == "__main__":
    max_x = 4
    max_y = 4
    env = PFenv(max_x, max_y)
    main(env)
'''

    env = PFenv(max_x, max_y)
    env.render()
    env.reset()
    env.render()
    env.step(1)
    env.render()
    done = False
    while not done:
        observation, reward, done, info = env.step(np.random.randint(0, 5))
        env.render() env.render()
'''
