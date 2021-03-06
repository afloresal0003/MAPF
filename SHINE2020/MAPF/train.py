import torch
import numpy as np
from models import *
from finalMAPFalgo import *
from collections import namedtuple
from itertools import count
from torch.distributions import Categorical
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from gifMaker import *

# Maps from integer actions to directions
action_dict = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}

# variables in named tuples can be referenced with dot syntax, like s.log_prob or s.value
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def select_actions(model, obs_n):
    """
    Select actions chooses n actions given n observations with the given model
    """
    # Feed the observations to the model, get action probabilities and state values for each agent
    probs_n, state_value_n = model(obs_n)

    # create a categorical distribution over the list of probabilities of actions
    m_n = [Categorical(probs) for probs in probs_n]

    # and sample an action using the distribution for each agent
    actions_n = [m.sample() for m in m_n]

    # save to action buffer
    model.saved_actions.append([SavedAction(m.log_prob(action), state_value) for m, action, state_value in zip(m_n, actions_n, state_value_n)])

    # return a list of actions to take, we just want an integer action, not a tensor and gradient, so we call .item() to get the value of the tensor
    return [action.item() for action in actions_n]

def finish_episode(model, optimizer, n_agents, eps=.0001):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    # We will keep track of the returns R (total reward for an entire episode) for each agent
    R = np.zeros(n_agents)

    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values
    gamma = 0.995

    # calculate the true value using rewards returned from the environment
    # R is the total returns (cumulative reward) through a certain time step
    # We are making a list 'returns' that will hold the total returns for each time step
    # As in returns[5] will hold the total reward received through timestep 5 (for each agent)
    # Returns will be a t x n array, that's t items long for each timestep
    # and each item holds an array of n items long for n agents
    for r in model.rewards[::-1]: # iterating through backwards (::-1)
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    # Turn into tensor and normalize, normalizing rewards/inputs typically helps NNs learn
    returns = torch.tensor(returns).float()
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # Calculate policy and value loss for each action, each agent
    for  i in range(len(saved_actions)):
        for j in range(n_agents):
            (log_probs, values) = saved_actions[i][j]
            R = returns[i][j]
            advantages = R - values.item()
            # calculate actor (policy) loss
            policy_losses.append(-log_probs * advantages)
            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(values, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    # Since we have one model, we can combine all the losses together from each agent
    # In some multi-agent RL, you have a different model for each agent, in which case you just wouldn't sum here, you'd keep them apart
    loss = torch.stack(policy_losses).sum(axis=-1) + torch.stack(value_losses).sum(axis=-1)

    # Calculate gradient (how the output changes with respect to the input)
    loss.backward()

    # Code for clipping gradients, norm of a vector essentially is the sqr root of the sum of the squares of each value (like distance formula)
    # Sometimes gradients get quite large and will lead to large steps, which might cause chaos, so we can restrict gradients to be between a min and max (called clipping)
    # nn.utils.clip_grad_norm_(model.parameters(), 1)

    # Step in direction of gradient (step in direction of improvement)
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

    # Return losses just for logging
    return (torch.stack(policy_losses).mean().detach()), (torch.stack(value_losses).mean().detach())


def train(model, env):
    writer = SummaryWriter()
    # Completely guessed on LR, might need to be different
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    running_reward = 10
    log_interval = 50
    video_interval = 10000
    save_interval = 1000
    save_directory = 'models/fixed_observation_'
    n_agents = env.n_agents
    running_length = 49
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        env.reset()
        obs_n = None
        ep_reward = 0
        ep_length = 0
        frames = []

        # Render every few thousand episodes
        render = i_episode % video_interval == 0
        if render:
            frames.append(np.rollaxis(env.render(),2, 0))

        for t in range(1, 50):
            ep_length += 1

            # select action from policy
            if obs_n is None:
                obs_n = [env.get_observation(i) for i in range(n_agents)]
            actions = select_actions(model, obs_n)

            # take the action
            obs_n, rewards, done, info = env.step(actions)

            if render:
                # Roll axis to make channels first for saving with tensorboard
                # tensorboard wants C x W x H, but to make gifs with matplotlib or imageio you need W x H x C
                # tensorboard is weird, most RGB data is W x H x C I believe
                frames.append(np.rollaxis(env.render(),2, 0))

                #frames.append(env.render())

            # Keep track of rewards
            model.rewards.append(rewards)
            ep_reward += np.sum(rewards)

            # Break the loop if episode is done, close render if applicable
            if done:
                if render:
                    env.render(close=True)
                break

        # update cumulative reward and length
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        running_length = 0.05 * ep_length + (1 - 0.05) * running_length

        # perform backprop, update model
        p_loss, v_loss = finish_episode(model, optimizer, n_agents)

        # log results
        # log the last reward received, the running average reward, policy loss, value loss, and running average length of episode
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            writer.add_scalar("Reward", ep_reward, i_episode)
            writer.add_scalar("Running Reward", running_reward, i_episode)
            writer.add_scalar("P_Loss", p_loss, i_episode)
            writer.add_scalar("V_Loss", v_loss, i_episode)
            writer.add_scalar("length", running_length, i_episode)

        # Save gifs to tensorboard, time consuming video process, use sparingly (>1000 iterations)
        if i_episode % video_interval == 0:
            writer.add_video("training_{}_vid".format(i_episode), torch.tensor(frames).unsqueeze(0), i_episode)
            # save_gif(frames, 'training_{}.gif'.format(i_episode)) # Unroll frames if you want to save it locally

        # Save the model every so often, since this while loop never ends on its own
        if i_episode % save_interval == 0:
            print('saving model to... ', save_directory + str(i_episode) + '.pt')
            torch.save(model.state_dict(), save_directory + str(i_episode) + '.pt')

        # break for Profiling
        # if i_episode == 2000:
        #    break
    env.close()

if __name__ == "__main__":
    # Initialize environment, we can keep one env object, but change the size and n_agents before resetting
    env = MAPFenv(4, 4, 3, prob_obs=0, observation_size=7)

    # Initialize model
    model = ConvNet(in_channels=4, height=7, width=7)
    # model = DeepConvNet()
    # model = Primal()

    # Train model
    train(model, env)

    # Profiling code, make sure to make train exit after some number of iterations (2000 or so maybe)
    # Gives a detailed report of runtime for each method/piece of code
    # Used to track down slow methods that we can optimize
    # Copy this code whenever you need it, just change what's between enable and disable to change what you time
    """
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    train(model, env)
    pr.disable()
    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(50)
    print(s.getvalue())
    """
