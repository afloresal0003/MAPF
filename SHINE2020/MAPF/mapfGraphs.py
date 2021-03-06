import numpy as np
import torch
import matplotlib.pyplot as plt
from finalMAPFalgo import MAPFenv
from train import select_actions
from models import *
from gifMaker import *

def vary_size_experiment(env, model):
    n_trials = 20
    solution_length = []
    for i in range(3, 20):
        # We'll put i agents into an ixi sized grid, see how long it takes to complete on average
        env.num_agents = i
        env.max_x = i
        env.max_y = i

        solution_length.append([])
        # Run n trials to get an average
        for j in range(n_trials):
            length, _ = run_episode(env, model, max_length = 250)
            solution_length[-1].append(length)

    plt.plot(np.mean(solution_length, axis=1))
    plt.title("Solution Length per increase in width")
    plt.xlabel("X-Axis")
    plt.ylabel("Solution Length")
    plt.show()

def new_experiment(env,model):
    n_trials = 25
    num_steps = []
    start = 3
    end = 25
    success_r = []
    x =[]
    for i in range(start, end):
        # We'll put i agents into an ixi sized grid, see how long it takes to complete on average
        env.max_x = i
        env.max_y = i
        x.append(i)
        num_steps.append([])
        success_r.append([])
        # Run n trials to get an average

        for j in range(n_trials):
            length, frames = run_episode(env, model, max_length = 50, render=False)
            if env.done() == True:
                success_r[-1].append(1)
            else:
                success_r[-1].append(0)
            num_steps[-1].append(env.numSteps)
            #save_gif(frames, "{}_{}_{}_{}.gif".format(i, i, i, j))
    plt.plot(x, np.mean(num_steps, axis=1)/np.arange(start,end))
    std = np.std(num_steps, axis=1) / np.arange(start, end)
    plt.fill_between(np.arange(end-start), np.mean(num_steps, axis=1)/np.arange(start,end) - std, np.mean(num_steps, axis=1)/np.arange(start,end) + std, alpha=0.2)
    #plt.savefig("numSteps.jpeg")
    plt.show()

    plt.plot(x, np.mean(success_r, axis=1))
    #plt.savefig("successRate.jpeg")
    plt.title("Success Rate Per Increase in Number of Agents")
    plt.xlabel("X-Axis")
    plt.ylabel("Success Rate")
    plt.show()

def vary_other_experiment(env,model):
    n_trials = 20
    solution_length = []
    x =[]
    for i in range(3, 20):
        # We'll put i agents into an ixi sized grid, see how long it takes to complete on average
        env.max_x = i
        env.max_y = i
        x.append(i)

        solution_length.append([])
        # Run n trials to get an average
        for j in range(n_trials):
            length, _ = run_episode(env, model, max_length = 250)
            solution_length[-1].append(length)

    plt.plot(x, np.mean(solution_length, axis=1))
    plt.title("Solution Length per increase in size of environment")
    plt.xlabel("Size")
    plt.ylabel("Solution Length")
    plt.show()

def vary_obs_experiment(env,model):
    n_trials = 20
    solution_length = []
    x =[]
    start
    for i in range(3, 20):
        # We'll put i agents into an ixi sized grid, see how long it takes to complete on average
        env.prob_obs = i / 50
        x.append(i/50)

        solution_length.append([])
        # Run n trials to get an average
        for j in range(n_trials):
            length, _ = run_episode(env, model, max_length = 250)
            solution_length[-1].append(length)

    plt.plot(x, np.mean(solution_length, axis=1))
    plt.title("Solution Length per increase in size of environment")
    plt.xlabel("Size")
    plt.ylabel("Solution Length")
    plt.show()

def run_episode(env, model, render=False, max_length=50):

    env.reset()
    obs_n = None
    ep_length = 0
    n_agents = env.n_agents

    frames = []

    if render:
        frames.append(env.render())

    for t in range(max_length):
        if obs_n is None:
            obs_n = [env.get_observation(i) for i in range(n_agents)]
        actions = select_actions(model, obs_n)

        obs_n, rewards, done, info = env.step(actions)

        if render:
            frames.append(env.render())

        ep_length += 1
        if done:
            if render:
                env.render(close=True)
            break

    # Modify to include anything you wish to measure
    return ep_length, frames

def load_model(path, model):
    """
    Loads a trained model
    Note that you need to already make a model object of the same type as the saved model
    loading the state dict essentially assigns all the weights and biases in the network, but you need to know how many/what they are by creating a model object already
    """
    model.load_state_dict(torch.load(path))
    return model

if __name__ == '__main__':
    env = MAPFenv(4, 4, 3, prob_obs=0.0, observation_size=7)
    model = ConvNet(in_channels=4, height=7, width=7)
    path = 'models/dense_600000.pt'
    model = load_model(path, model)
    new_experiment(env, model)
