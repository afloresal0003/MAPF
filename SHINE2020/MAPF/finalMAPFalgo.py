import numpy as np
import gym
import random
from matplotlib.colors import hsv_to_rgb
from gym.envs.classic_control import rendering
import math
action_dict = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
#1 work on observations
#2 create a NN
#3 write an environment loop, make it run
#4 Training
class MAPFenv(gym.Env):
    """
    A Multi-Agent Pathfinding environment

    The task is to navigate agents from start positions to goal positions
    without colliding with each other.
    """

    def __init__(self, max_x, max_y, n_agents=5, prob_obs = 0.1, observation_size=11):
        """
        max_x, max_y are the bounds of the world
        n_agents is the number of agents placed in the world
        prob_obs is the probability an obstacle is placed at any given position
        observation_size is the size of the observation to be returned
        """

        self.max_x = max_x
        self.max_y = max_y
        self.observation_size = observation_size
        self.prob_obs = prob_obs
        self.n_agents = n_agents
        self.state = np.zeros((3, self.max_x, self.max_y))
        self.obstacles = []
        self.reset()
        self.viewer = None
        self.numSteps = 0

        #Guarantees you can solve an environment
    def check_connection(self):
        # Implementation of depth first search
        # Find an empty cell to start at
        x, y = 0, 0
        visited_cells = []
        to_explore = []

        # Find first empty cell
        empty = False
        for i in range(len(self.state[0])):
            for j in range(len(self.state[1])):
                if self.state[0, i, j] == 0:
                    empty = True
                    break
            if empty:
                x, y = i, j
                break
        # No empty cells...
        if empty == False:
            return False
        to_explore.append((x, y))
        # x, y is our starting point, try and visit all points
        while True:
            #print(len(to_explore))
            if len(to_explore) == 0:
                break
            a, b = to_explore.pop()
            #print('exploring', a, b, len(to_explore))
            if (a,b) not in visited_cells:
                visited_cells.append((a,b))
            if self.valid_coordinates(a + 1, b):
                if (a + 1, b) not in visited_cells and self.state[0, a+1, b] == 0:
                    to_explore.append((a+1, b))
            if self.valid_coordinates(a - 1, b):
                if (a - 1, b) not in visited_cells and self.state[0, a - 1, b] == 0:
                    to_explore.append((a-1, b))
            if self.valid_coordinates(a, b + 1):
                if (a, b+1) not in visited_cells and self.state[0, a, b+1] == 0:
                    to_explore.append((a, b+1))
            if self.valid_coordinates(a, b-1):
                if (a, b-1) not in visited_cells and self.state[0, a, b-1] == 0:
                    to_explore.append((a, b-1))

        num_obstacles = np.count_nonzero(self.state[0])
        num_connected_free = len(visited_cells)

        if num_obstacles + num_connected_free == np.size(self.state[0]):
            #print('connected', self.state[0])
            return True
        else:
            #print(num_obstacles)
            #print(num_connected_free)
            #print(np.size(self.state[0]))
            #print("Not connected: ", self.state[0])
            return False

    def valid_coordinates(self, x, y):
        if x >= 0 and x < self.max_x and y >= 0 and y < self.max_y:
            return True
        else:
            return False

    def set_obstacles(self):
        connected = False
        while not connected:
            self.obstacles = []
            self.state[0] = np.zeros_like(self.state[0])
            for row in range(len(self.state[0])):
                for i in range(len(self.state[0][0])):
                    if (random.random()) < self.prob_obs:
                        self.state[0, row, i] = -1
                        self.obstacles.append((row, i))
            connected = self.check_connection()
            if np.count_nonzero(self.state[0]) + self.n_agents >= self.max_x * self.max_y:
                connected = False
                # Not possible to place agents


    def reset(self):
        self.numSteps = 0
        # state is a 3xwxh array, where the first channel contains the obstacles, the second channel the agents, and the third channel the goals
        self.state = np.zeros((3, self.max_x, self.max_y))
        self.set_obstacles()
        # Put the agents into a list
        # Put the goals into a list
        self.agents = []
        self.goals = []
        for n in range(self.n_agents):
            a = (np.random.randint(self.max_x),
                np.random.randint(self.max_y))
            # Generate locations until the agent is not on an obstacle
            while a in self.agents or self.state[0, a[0], a[1]] == -1:
                a = (np.random.randint(self.max_x),
                        np.random.randint(self.max_y))
            self.agents.append(a)
            # Put the agent into the state array (+1 so we don't put 0 into an array of zeros)
            self.state[1, a[0], a[1]] = n + 1

            #Do the same for goals
            g = (np.random.randint(self.max_x),
                    np.random.randint(self.max_y))
            while g in self.goals or self.state[0, g[0], g[1]] == -1:
                g = (np.random.randint(self.max_x),
                        np.random.randint(self.max_y))
            self.goals.append(g)
            self.state[2, g[0], g[1]] = n + 1
        self.goals = np.array(self.goals)
        self.agents = np.array(self.agents)
        self.obstacles = np.array(self.obstacles)
        return

    def render(self, mode='human', close=False, screen_width=800, screen_height=800):
        if close == True:
            return
        if mode == 'ANSI':
            print(self.state)
        elif mode == 'human':
            size = screen_width / max(self.max_x, self.max_y)
            colors = self.init_colors()

            # If we have no viewer, create one
            if self.viewer is None:
                self.viewer = rendering.Viewer(screen_width, screen_height)
                self.reset_renderer = True
            # if self.reset_renderer:
            if True:
                self.create_rectangle(0, 0, screen_width, screen_height, (.6, .6, .6), permanent=True)
                for i in range(self.max_x):
                    start = 0
                    end = 1
                    scanning = False
                    write = False
                    for j in range(self.max_y):
                        if self.state[0, i, j] != -1 and not scanning:
                            start = j
                            scanning = True
                        if (j == self.max_y - 1 or self.state[0, i, j] == -1) and scanning:
                            if j == self.max_y - 1:
                                end = j+1
                            else:
                                end = j
                            scanning = False
                            write = True
                        if write:
                            x = i * size
                            y = start * size
                            self.create_rectangle(x, y, size, size*(end-start), (1, 1, 1), permanent=True)
                            write = False

            for agent_id, a in enumerate(self.agents):
                i, j = a
                x, y = i * size, j * size
                color = colors[agent_id + 1]
                self.create_rectangle(x, y, size, size, color)

                # Goals
                i, j = self.goals[agent_id]
                x, y = i * size, j * size
                self.create_circle(x, y, size, size, color)

                if (self.agents[agent_id] == self.goals[agent_id]).all():
                    color = (0, 0, 0)
                    self.create_circle(x, y, size, size, color)

            self.reset_renderer = False
            result = self.viewer.render(return_rgb_array=True)

            return result

    def drawStar(self, centerX, centerY, diameter, numPoints, color):
        """
        Draw's a star, not currently used
        """
        outerRad=diameter//2
        innerRad=int(outerRad*3/8)
        #fill the center of the star
        angleBetween=2*math.pi/numPoints#angle between star points in radians
        for i in range(numPoints):
            #p1 and p3 are on the inner radius, and p2 is the point
            pointAngle=math.pi/2+i*angleBetween
            p1X=centerX+innerRad*math.cos(pointAngle-angleBetween/2)
            p1Y=centerY-innerRad*math.sin(pointAngle-angleBetween/2)
            p2X=centerX+outerRad*math.cos(pointAngle)
            p2Y=centerY-outerRad*math.sin(pointAngle)
            p3X=centerX+innerRad*math.cos(pointAngle+angleBetween/2)
            p3Y=centerY-innerRad*math.sin(pointAngle+angleBetween/2)
            #draw the triangle for each tip.
            poly=rendering.FilledPolygon([(p1X,p1Y),(p2X,p2Y),(p3X,p3Y)])
            poly.set_color(color[0],color[1],color[2])
            poly.add_attr(rendering.Transform())
            self.viewer.add_onetime(poly)

    def create_rectangle(self,x,y,width,height,fill,permanent=False):
        """
        Draws a rectangle
        """
        ps=[(x,y),((x+width),y),((x+width),(y+height)),(x,(y+height))]
        rect=rendering.FilledPolygon(ps)
        rect.set_color(fill[0],fill[1],fill[2])
        rect.add_attr(rendering.Transform())
        if permanent:
            self.viewer.add_geom(rect)
        else:
            self.viewer.add_onetime(rect)

    def create_circle(self,x,y,diameter,size,fill,resolution=20):
        """
        Draws a circle
        """
        c=(x+size/2,y+size/2)
        dr=math.pi*2/resolution
        ps=[]
        for i in range(resolution):
            x=c[0]+math.cos(i*dr)*diameter/2
            y=c[1]+math.sin(i*dr)*diameter/2
            ps.append((x,y))
        circ=rendering.FilledPolygon(ps)
        circ.set_color(fill[0],fill[1],fill[2])
        circ.add_attr(rendering.Transform())
        self.viewer.add_onetime(circ)

    def init_colors(self):
        """
        Creates a unique color for each agent (not actually unique though)
        """
        c={a+1:hsv_to_rgb(np.array([a/float(self.n_agents),1,1])) for a in range(self.n_agents)}
        return c


    def get_observation(self, agent_id):
        """
        Get observation for agent_id
        Observation is a local view of the world, centralized around the agent
        It contains a piece of image data, and a vector to the goal
        the image data contains 4 channels, (obstacles, agents, goals, agent's goal)
        """
        rel_positions = []
        goal_positions = []
        # Calculate relative positions of all agents and all goals to the central agent
        for i in range(len(self.agents)):
             rel_positions.append(self.agents[i] - self.agents[agent_id])
             goal_positions.append(self.goals[i]-self.agents[agent_id])

        agents = np.zeros((self.observation_size, self.observation_size))
        goals = np.zeros_like(agents)
        obstacles = np.zeros_like(agents)
        center = self.observation_size // 2

        # Check if other agents are within the field of view
        for i, j in rel_positions:
            if i + center < 0 or j + center<0:
                continue
            if i+center >= self.observation_size or j + center >= self.observation_size:
                continue
            else:
                agents[i + center, j + center] = 1

        # Repeat for goals
        for i, j in goal_positions:
            if i+center < 0 or j+center<0:
                continue
            if i + center>= self.observation_size or j+center >= self.observation_size:
                continue
            else:
                goals[i+center, j+center] = 1
       # Calculate for each cell in our local view of the obstacles, if an obstacle exists in the world, or if the local coordinate is not in the global world (out of bounds, a wall)
        for i in range(len(obstacles)):
            for j in range(len(obstacles[i])):
                global_i, global_j = i - center - self.agents[agent_id][0], j - center - self.agents[agent_id][1]
                if global_i < 0 or global_j <0 or global_i >= self.state.shape[1] or global_j >= self.state.shape[2]:
                    obstacles[i, j] = 1
                elif self.state[0, global_i, global_j] == -1:
                    obstacles[i, j] = 1

        # Add in a layer that contains the location of our own goal for agent_id
        # If not in FOV, map to the edge of it. So if you can't see your own goal, it'll be placed in the direction of your goal on the edge of your observation
        own_goal = np.zeros_like(goals)
        goal_pos = [self.goals[agent_id][0] - self.agents[agent_id][0], self.goals[agent_id][1] - self.agents[agent_id][1]]
        if goal_pos[0] < 0:
            goal_pos[0] = 0
        elif goal_pos[0] >= self.observation_size:
            goal_pos[0] = self.observation_size - 1
        if goal_pos[1] < 0:
            goal_pos[1] = 1
        elif goal_pos[1] >= self.observation_size:
            goal_pos[1] = self.observation_size - 1
        own_goal[goal_pos[0], goal_pos[1]] = 1

        # Goal vector is the relative dx and dy to the goal from the agent
        goal_vector = self.goals[agent_id] - self.agents[agent_id]

        return (np.stack((obstacles, agents, goals, own_goal)), goal_vector)


    def step(self, actions):
        """
        Step takes a list of actions, one for each agent, and tries to execute those actions
        step returns observations and rewards for each agent, and whether or not the environment is solved (done)
        The final return as None, represents the info return that gym uses to include other information
        you can use this slot to return for instance, how many agents are on their goal, how many agents failed to execute their action (chose an invalid action), or any other information you so choose
        """
        rewards = []
        # Execute individual steps, keep track of individual rewards
        for n in range(self.n_agents):
            rewards.append(self._step(actions[n],n))

        # If finished, add additional reward for each agent
        if self.done() == True:
            # print("Finished!")
            done = True
            # Big reward for all agents finishing, smaller reward for single agent finishing
            for i in range(len(rewards)):
                rewards[i] += 20
        else:
            done = False
        # Per agent observations
        obs = [self.get_observation(n) for n in range(self.n_agents)]

        # observation is a list, rewards are also a list, n items long, one for each agent
        return obs, rewards, done, None

    def _step(self, action, agent_id):
        """
        takes a single action and a single agent, attempts to execute that action
        returns a reward for that single agent
        """

        # Grab direction from the action dictionary
        direction = action_dict[action]

        # If everyone's done, let's not screw it up by taking an action
        if self.done():
            return 0
        # If it's a valid action, execute it by moving the agent in self.state and the value in self.agents

        if self.is_valid_action(action, agent_id) == True:
            if action != 0:
                self.numSteps +=1
            self.state[1, self.agents[agent_id][0], self.agents[agent_id][1]] = 0
            self.agents[agent_id] += np.array(direction)
            self.state[1, self.agents[agent_id][0], self.agents[agent_id][1]] = agent_id + 1
            reward = 0
        # If not a valid action, don't do anything, get negative reward
        else:
            reward = -.3

        # If agent on goal, get small positive reward...
        # This is very dangerous... but it seems to work.
        # It's dangerous because in theory, it could be better to have n-1
        # agents sit on their goal for the entire episode and not finish,
        # because not finishing might be a higher reward than finishing
        if self.agents[agent_id][0] == self.goals[agent_id][0] and self.agents[agent_id][1] == self.goals[agent_id][1]:
            reward = 1
        # If you're not done, get a penalty. Higher penalty for staying still, encourages exploration
        else:
            done = False
            if action == 0:
                reward += -.5
            else:
                reward += -0.3
        return reward

    def is_valid_action(self, action, agent_id):
        """
        Checks if the given action is valid for a given agent, checks collisions with obstacles, walls, and other agents
        """
        direction = action_dict[action]
        loc = self.agents[agent_id] + np.array(direction)
        if loc[0] >= 0 and loc[0] < self.max_x:
            if loc[1] >= 0 and loc[1] < self.max_y:
                if self.state[0, loc[0], loc[1]] != -1:
                    if self.state[1, loc[0], loc[1]] == 0:
                        return True
        return False

    def done(self):
        """
        Checks whether all agents are on their goals
        """
        for a, g in zip(self.agents, self.goals):
            if a[0] != g[0] or a[1] != g[1]:
                return False
        return True

    def close(self):
        # Close method for rendering
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    env = MAPFenv(4, 4, 5, prob_obs=.7)
    env.render()

    actions = [0,1,2,3,4]
    print(env.step(actions))

    if env.done():
        print("FINISHED!")
    env.reset()
