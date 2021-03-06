import numpy as np
import gym
import random
action_dict = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
#1 work on observations
#2 create a NN
#3 write an environment loop, make it run
#4 Training


class MAPFenv(gym.Env):
    def __init__(self, max_x, max_y, n_agents=5, prob_obs = 0.1):
        self.max_x = max_x
        self.max_y = max_y
        self.prob_obs = prob_obs
        self.n_agents = n_agents

        self.state = np.zeros((3, self.max_x, self.max_y))

        for row in range(len(self.state[0])):
            for i in range(len(self.state[0][0])):
                if (random.random()) < prob_obs:
                    self.state[0, row, i] = -1

        # Put the agents into a list
        # Put the goals into a list
        self.agents = []
        self.goals = []
        for n in range(self.n_agents):
            a = (np.random.randint(self.max_x),
                np.random.randint(self.max_y))
            # while self.state[1, a[0], a[1]] != 0:
            while a in self.agents or self.state[0, a[0], a[1]] == -1:
                a = (np.random.randint(self.max_x),
                        np.random.randint(self.max_y))

            self.agents.append(a)
            #Question: What does this line actually do?
            self.state[1, a[0], a[1]] = n + 1

            g = (np.random.randint(self.max_x),
                    np.random.randint(self.max_y))
            while g in self.goals or self.state[0, g[0], g[1]] == -1:
                g = (np.random.randint(self.max_x),
                        np.random.randint(self.max_y))
            self.goals.append(g)
            self.state[2, g[0], g[1]] = n + 1

        # Agents are odd numbers, goals are even?
        # state into 3d array (3, w, h)? (obstacles, agents, goals)?

    def reset(self):
        self.state = np.zeros((3, self.max_x, self.max_y))
        for row in range(len(self.state[0])):
            for i in range(len(self.state[0][0])):
                if (random.random()) < self.prob_obs:
                    self.state[0, row, i] = -1

        # Put the agents into a list
        # Put the goals into a list
        self.agents = []
        self.goals = []
        for n in range(self.n_agents):
            a = (np.random.randint(self.max_x),
                np.random.randint(self.max_y))
            # while self.state[1, a[0], a[1]] != 0:
            while a in self.agents or self.state[0, a[0], a[1]] == -1:
                a = (np.random.randint(self.max_x),
                        np.random.randint(self.max_y))

            self.agents.append(a)
            self.state[1, a[0], a[1]] = n + 1

            g = (np.random.randint(self.max_x),
                    np.random.randint(self.max_y))
            while g in self.goals or self.state[0, g[0], g[1]] == -1:
                g = (np.random.randint(self.max_x),
                        np.random.randint(self.max_y))
            self.goals.append(g)
            self.state[2, g[0], g[1]] = n + 1
        return self.get_observation(self.state)

    def render(self):
        print(self.state)

        #Where we left off
    def get_observation(self, state, agent_id):
        '''obstacles = np.where(state==-1, 1, 0)
        goal = np.where(state==2, 1, 0)
        agent = np.where(state==1, 1, 0)
        return np.stack((obstacles, goal, agent))'''
        rel_positions = []
        goal_positions = []
        for i in range(len(self.agents)):
             rel_positions.append(self.agents[i] - self.agents[agent_id])
             goal_positions.append(self.goals[i]-self.agents[agent_id])

        agents = np.zeros((11,11))
        goals = np.zeros((11,11))
        obstacles = np.zeros((11,11))

        for i, j in rel_positions:
            if i+5 < 0 or j+5<0:
                continue
            if i+5 >= 11 or j+5 >= 11:
                continue
            else:
                agents[i+5, j+5] = 1

        for i, j in goal_positions:
            if i+5 < 0 or j+5<0:
                continue
            if i+5 >= 11 or j+5 >= 11:
                continue
            else:
                goals[i+5, j+5] = 1

        for i in len(obstacles):
            for j in len(obstacles[i]):
                global_i, global_j = i - 5 - self.agents[agent_id][0], j - 5 - self.agents[agent_id][1]
                if global_i < 0 or global_j <0 or global_i >= self.state.shape[1] or global_j >= self.state.shape[2]:
                    obstacles[i, j] = 1
                elif self.state[0, global_i, global_j] == -1:
                    obstacles[i, j] = 1

        return np.stack((obstacles, goals, agents))

    # Multiple agents with different actions
    # Step takes in a list of actions
    def step(self, actions):
        rewards = []
        for n in range(self.n_agents):
            rewards.append(self._step(actions[n],n))
        if self.done() == True:
            done = True
            # Big reward for all agents finishing, smaller reward for single agent finishing
            for i in range(len(rewards)):
                rewards[i] += 50
        else:
            done = False
        # Per agent observations
        obs = [self.get_observation(self.state, n) for n in range(self.n_agents)]

        #obs = self.get_observation(self.state, agent_id)  # an x, y pair
        # observation is a list, rewards are also a list
        return obs, rewards, done, None

    def _step(self, action, agent_id):
        direction = action_dict[action]

        if self.is_valid_action(action, agent_id) == True:
            self.state[1, self.agents[agent_id][0], self.agents[agent_id][1]] = 0
            self.agents[agent_id] += np.array(direction)
            self.state[1, self.agents[agent_id][0], self.agents[agent_id][1]] = agent_id + 1
        if self.state[1, self.agents[agent_id][0], self.agents[agent_id][1]] == self.state[2, self.goals[agent_id][0], self.goals[agent_id][1]]:
            reward = 5
        else:
            done = False
            reward = -1
        return reward

    def is_valid_action(self, action, agent_id):
        direction = action_dict[action]
        loc = self.agents[agent_id] + np.array(direction)
        if loc[0] >= 0 and loc[0] < self.max_x:
            if loc[1] >= 0 and loc[1] < self.max_y:
                if self.state[0, loc[0], loc[1]] != -1:
                    if self.state[1, loc[0], loc[1]] == 0:
                        return True
        return False

    def done(self):
        for n in range(self.n_agents):
            if self.state[1, self.agents[n][0], self.agents[n][1]] != self.state[2, self.goals[n][0], self.goals[n][1]]:
                return False
        return True

    def close(self):
        pass


if __name__ == "__main__":
    env = MAPFenv(4, 4, 5)
    env.render()
    actions = [0,1,2,3,4]
    env.step(actions)
    #env.reset()
