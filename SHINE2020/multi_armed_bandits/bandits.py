import numpy as np
import matplotlib.pyplot as plt

class Bandits:
    def __init__(self, n=10):
        self._bandits = [(np.random.randint(10), np.random.randint(10)) for i in range(n)]
        self.history = []
        self.previous_experiments = []
        self.labels = []

    def end_experiment(self, label):
        self.previous_experiments.append(self.history)
        self.history = []
        self.labels.append(label)

    def sample_bandit(self, k):
        result = np.random.normal(*self._bandits[k])
        self.history.append(result)
        return result

    def plot_performance(self):
        for i, data in enumerate(self.previous_experiments):
            plt.plot(np.cumsum(data) / np.arange(1, len(data) + 1), label=self.labels[i])
        optimal_performance = np.ones_like(self.previous_experiments[0]) * np.max([x for x, y in self._bandits])
        plt.plot(np.cumsum(optimal_performance) / np.arange(1, len(optimal_performance) + 1), label='optimal')
        plt.xlabel('Num Iters')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    n = 10
    bandits = Bandits(n)
    num_iters = 1000

    # Uniform random
    for t in range(num_iters):
        bandits.sample_bandit(np.random.randint(n))
    bandits.end_experiment('Uniform Random')

    # Rotating
    for t in range(num_iters):
        bandits.sample_bandit(t % n)
    bandits.end_experiment('Rotating')

    results = [[] for i in range(n)]



    for t in range(num_iters):
        if t > 100:
            k = np.argmax([np.mean(results[i]) for i in range(len(results))])
        else:
            k = np.random.randint(n)
        result = bandits.sample_bandit(k)
        results[k].append(result)
    bandits.end_experiment("Threshold")

    means = np.zeros(n)
    counts = np.zeros(n)
    for t in range(num_iters):
        if t > 100:
            k = np.argmax(means)
        else:
            k = np.random.randint(n)
        result = bandits.sample_bandit(k)
        means[k] = (means[k] * counts[k] + result) / (counts[k] + 1)
        counts[k] += 1
    bandits.end_experiment("Faster Mean")

    means = np.zeros(n)
    counts = np.zeros(n)
    epsilon = .1
    for t in range(num_iters):
        if np.random.rand() > epsilon:
           k = np.argmax(means)
        else:
            k = np.random.randint(n)
        result = bandits.sample_bandit(k)
        means[k] = (means[k] * counts[k] + result) / (counts[k] + 1)
        counts[k] += 1
    bandits.end_experiment("Epsilon Greedy")

    bandits.plot_performance()
