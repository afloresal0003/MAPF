import numpy as np

def make_gifs(world, goals, paths, run_name, environment_name, frame_length=0.1):
    from mapf_gym import MAPFEnv
    import imageio
    num_agents = np.max(world)
    env = MAPFEnv(num_agents=num_agents, world0=world, goals0=goals)
    dirDict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}
    frames = []
    for step_num in range(1, len(paths)):
        for agent in range(len(paths[step_num])):
            action = dirDict[tuple(paths[step_num, agent] - paths[step_num - 1, agent])]
            env.step((agent + 1, action))
        results = env._render(mode='rgb_array')
        frames.append(results)
    images = np.array(frames)
    imageio.mimwrite('GIFS/{}_{}_.gif'.format(run_name, environment_name), images, subrectangles=True,
                     duration=len(images) * frame_length)
    print("wrote gif 'GIFS/{}_{}_.gif'".format(run_name, environment_name))

def save_gif(frames, file_name):
    import imageio
    frame_length = .01
    images = np.array(frames)
    imageio.mimwrite('GIFS/{}'.format(file_name), images, subrectangles=True,
                     duration=len(images) * frame_length)
    print("wrote gif 'GIFS/{}'".format(file_name))
