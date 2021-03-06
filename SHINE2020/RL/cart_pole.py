import gym
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import random
env = gym.make('CartPole-v0')
from torch.utils.tensorboard import SummaryWriter


class NN(nn.Module):
    """
    The Neural Network that will convert from observations to q-values
    """

    def __init__(self, input_shape, output_shape, n_hidden=20, n_layers=2):
        # Super must be called for classes that inherit from nn.Module
        # Super runs the constructor of the parent class
        super(NN, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        # A Sequential block contains a series of operations that are run sequentially
        # Linear layers are the most common layer we'll use
        # Every layer except the last must be followed by a non-linear operation
        # ReLU is 0 if the input is < 0, and is the input if the input is > 0
        self.net = nn.Sequential(nn.Linear(input_shape, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, output_shape),
                            )

    def forward(self, x):
        """
        The forward method runs the neural network with input x
        It returns the output of the network, which in this case are the q-values
        Forward can be called by model(x) or model.forward(x), they are equivalent
        You can put in other operations than just running the sequential,
        For instance, we'll see some networks with multiple outputs that are returned
        """
        return self.net(x)

class ReplayBuffer:

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self._buffer = []

    def append(self, experience):
        self._buffer.append(experience)

    def get_buffer(self):
        random.shuffle(self._buffer)
        while len(self._buffer) > self.max_size:
            self._buffer.pop()
        return self._buffer



def train(model, env):
    writer = SummaryWriter()
    replay_buffer = ReplayBuffer()
    gamma = .99
    learning_rate = 10e-3
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    epsilon = .1
    for episode in range(10000):
        observation = env.reset()
        done = False
        num_steps = 0
        # Decay our exploration rate
        epsilon *= epsilon_decay

        # Interact with environment until done
        while not done:
            # gym gives us numpy arrays, but for pytorch we need tensors, float avoids a float/double disagreement
            observation_tensor = torch.tensor(observation).float()
            # Run the model, get the q-values
            q_values = model(observation_tensor)

            # balance exploration/exploitation
            if np.random.rand() > max(minimum_epsilon, epsilon):
                action = torch.argmax(q_values)
                # again, gym needs numpy, so we need to convert actions to numpy
                action = action.detach().numpy()
            else:
                # Exploration
                action = env.action_space.sample()

            # We need to keep track of the old state, the action we took,
            # the reward we got, and the new state we entered
            old_observation = observation
            # Take the action, get information from the environment
            observation, reward, done, info = env.step(action)
            num_steps += 1
            # Store our information for later
            replay_buffer.append((old_observation, action, reward, observation))

        steps.append(num_steps)

        # Update the NN every so often
        if episode % update_frequency == 0:
            # Reset the optimizer and loss
            optimizer.zero_grad()
            loss = 0

            # Get a batch from the replay buffer and loop through it
            for (state, action, reward, new_state) in replay_buffer.get_buffer(batch_size):
                """
                Q-values are defined to be Q(s, a) = r + gamma * max Q(s', a'),
                where s is the initial state (observation), a is the action taken,
                r is the reward received, and s' is the next state (after a step is taken)

                Before training, our network doesn't output correct q-values,
                we can train the network by training to make the equality hold as follows:

                1. Gather an experience (s, a, r, s') from the batch
                2. Calculate the left hand side (LHS): Q(s, a), which is the neural network output at the given action index
                3. Calculate the right hand side (RHS): r + gamma max Q(s', a'), by running the NN on the next state and taking the maximum output
                4. Compute Loss = (LHS - RHS)^2, which will eventually force LHS=RHS
                5. When LHS=RHS, we have converged (but this needs to hold for every state not just one!)
                """
                q_values = model(torch.tensor(state).float())
                q = q_values[action]
                y = reward + gamma * torch.max(model(torch.tensor(new_state).float()))
                loss += (q-y)**2

            # Run the update
            loss.backward()
            optimizer.step()

            # Add values to our tensorboard graphs
            writer.add_scalar('Steps', num_steps, episode)
            writer.add_scalar('Loss', loss.detach().item(), episode)

        # Every so often, let's just check how we do without exploration messing with us
        if episode % 10 == 0:
            observation = env.reset()
            done = False
            t = 0
            while not done:
                observation_tensor = torch.tensor(observation).float()
                q_values = model(observation_tensor)
                action = torch.argmax(q_values)
                action = action.detach().numpy()
                observation, reward, done, info = env.step(action)
                t += 1
            print("Episode: {} | average reward (last 10 trials) {} | no-exploration length {}".format(episode, np.mean(steps[-10:]), t))

        # If we reach 100 steps 10 times in a row, we'll say we're done
        if (np.array(steps[-10:]) >= 100).all():
            print("Finished training...")
            break
    writer.close()

def save_model(model, filename):
    torch.save(model, filename)

def load_model(filename):
    model = torch.load(filename)
    return model

def run_visualization(model, env):
    """
    Run the environment and display the results
    """
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            time.sleep(.05)
            observation_tensor = torch.tensor(observation).float()
            q_values = model(observation_tensor)
            action = torch.argmax(q_values)
            # print("action", action)
            # env.action_space.sample()
            observation, reward, done, info = env.step(action.detach().numpy())
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

if __name__ == "__main__":
    use_saved_model = False
    saved_model = "model.pt"

    if use_saved_model:
        model = load_model(saved_model)
    else:
        model = NN(4, 2, n_hidden=12)
        train(model, env)

    # Make fun videos
    run_visualization(model, env)

    # You might not want to overwrite, but just change the filename
    save_model(model, saved_model)

    # Close out the environment (mostly for the displays)
    env.close()
