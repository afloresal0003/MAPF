import numpy as np
import gym

def get_action(model,observation):
    actions = np.matmul(model,observation)
    return np.argmax(actions)

def run_env(env,model,render=False):
    env.seed(1333)
    observation = env.reset()
    reward_sum = 0
    done = False
    while not done:
        if render:
            env.render()
        action = get_action(model,observation)

        observation,reward,done,info = env.step(action)
        reward_sum+=reward
    env.close()
    return reward_sum

def main():
    env = gym.make('LunarLander-v2')

    model_size = env.observation_space.shape[0] \
                 * env.action_space.n
    best_reward = -10 ** 18
    for i in range(2000):
        model = np.random.normal(size=model_size)

        model = np.reshape(model, (env.action_space.n, -1))

        reward = run_env(env,model)
        if reward > best_reward:
            print(reward)
            best_reward = reward
            run_env(env,model,render=True)

if __name__ == '__main__':
    main()
